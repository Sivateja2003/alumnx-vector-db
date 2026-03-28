from __future__ import annotations

import logging

import numpy as np

from app.config import get_config
from app.models import CandidateResult, QueryLogs, RetrieveRequest, RetrieveResponse
from app.services.embedding.embedder import GeminiEmbedder
from app.services.ingestion import EMBEDDABLE_SECTIONS, UNIVERSAL_VECTOR_STORE
from app.services.llm_query import classify_and_generate_sql
from app.services.store.postgres_store import PostgresStore
from app.services.store.vector_file_store import VectorFileStore

logger = logging.getLogger("nexvec.retrieval")


def retrieve_documents(request: RetrieveRequest) -> RetrieveResponse:
    """
    RDS-first retrieval with LLM-driven routing.

    Flow:
      1. LLM classifies the query → generates SQL + decides if vector search needed
      2. Execute SQL → Postgres returns matching resume_ids
      3a. RDS-only: fetch full resume rows, return ranked as SQL matched them
      3b. RDS + Vector: embed query → cosine similarity against SQL-filtered vectors
      4. Deduplicate by user_id — one result per person
      5. Return ranked candidates + query logs
    """
    config = get_config()
    k = request.k or config.knn_k
    pg = PostgresStore()

    if not request.query.strip():
        raise ValueError("EMPTY_QUERY")

    embedding_model = request.embedding_model or config.embedding_model
    logger.info("Retrieve: query=%r k=%s model=%s", request.query, k, embedding_model)

    # ── Step 1: Classify query → SQL + routing decision ───────────────
    sql_failed = False
    sql = ""
    needs_vector = False
    routing_reason = ""

    try:
        classification = classify_and_generate_sql(request.query)
        sql = classification.sql
        needs_vector = classification.needs_vector
        routing_reason = classification.reason
        resume_ids = pg.execute_sql_query(sql)
        logger.info(
            "SQL returned %d resume_ids | needs_vector=%s | reason=%s",
            len(resume_ids), needs_vector, routing_reason,
        )
    except Exception as exc:
        logger.warning("SQL classification/execution failed (%s), falling back to full scan", exc)
        resume_ids = []
        sql_failed = True
        routing_reason = f"SQL failed ({exc}), falling back to full scan with vector search"
        needs_vector = True

    sql_matched_count = len(resume_ids)

    if resume_ids:
        resume_rows = pg.get_resumes_by_ids(resume_ids)
    elif sql_failed:
        resume_rows = pg.get_all_active_resumes()
    elif needs_vector:
        # SQL keyword filter too narrow for a semantic query — cast a wider net.
        # Run vector search across all resumes instead of returning empty.
        logger.info(
            "SQL returned 0 for semantic query=%r — falling back to full-scan vector search",
            request.query,
        )
        routing_reason = routing_reason + " (SQL keyword too narrow — vector search on full pool)"
        resume_rows = pg.get_all_active_resumes()
    else:
        logger.info("SQL filter returned no matches for query=%r", request.query)
        logs = QueryLogs(
            user_query=request.query,
            sql_query=sql,
            sql_matched_count=0,
            routing_decision="rds_only",
            routing_reason=routing_reason,
            vector_search_used=False,
        )
        return RetrieveResponse(query=request.query, k_used=k, candidates=[], logs=logs)

    if not resume_rows:
        logger.info("No active resumes found")
        logs = QueryLogs(
            user_query=request.query,
            sql_query=sql,
            sql_matched_count=0,
            routing_decision="rds_only",
            routing_reason=routing_reason,
            vector_search_used=False,
        )
        return RetrieveResponse(query=request.query, k_used=k, candidates=[], logs=logs)

    resume_rows_by_id = {r["resume_id"]: r for r in resume_rows}

    # ── Step 2: RDS-only path (no vector search) ───────────────────────
    if not needs_vector:
        logger.info("RDS-only path: returning %d candidates without vector ranking", len(resume_rows))

        seen_users: set[str] = set()
        candidates: list[CandidateResult] = []

        # Preserve SQL result order (resume_ids is already ordered by RDS)
        ordered_ids = [rid for rid in resume_ids if rid in resume_rows_by_id]
        # Append any rows fetched but not in the ordered list (safety net)
        remaining = [r["resume_id"] for r in resume_rows if r["resume_id"] not in set(ordered_ids)]
        all_ordered = ordered_ids + remaining

        for rid in all_ordered:
            row = resume_rows_by_id.get(rid, {})
            uid = row.get("user_id", rid)
            if uid in seen_users:
                continue
            seen_users.add(uid)
            candidates.append(CandidateResult(
                user_id=uid,
                resume_id=rid,
                source_filename=row.get("source_filename", ""),
                similarity_score=None,
                name=row.get("name"),
                email=row.get("email"),
                phone=row.get("phone"),
                location=row.get("location"),
                work_experience_years=row.get("work_experience_years"),
                skills=list(row.get("skills") or []),
                objectives=row.get("objectives"),
                matched_sections=[],
                match_type="rds",
            ))
            if len(candidates) == k:
                break

        logs = QueryLogs(
            user_query=request.query,
            sql_query=sql,
            sql_matched_count=sql_matched_count,
            routing_decision="rds_only",
            routing_reason=routing_reason,
            vector_search_used=False,
        )
        logger.info("RDS-only complete: %d candidates", len(candidates))
        return RetrieveResponse(query=request.query, k_used=k, candidates=candidates, logs=logs)

    # ── Step 3: RDS + Vector path ──────────────────────────────────────
    logger.info("RDS + Vector path: scoring %d SQL-filtered resumes", len(resume_rows))

    # One vector per resume: work_experience_text if stored, else projects.
    # This matches the ingestion priority (EMBEDDABLE_SECTIONS order).
    chunk_to_resume: dict[str, str] = {}
    chunk_to_section: dict[str, str] = {}
    for row in resume_rows:
        for section in EMBEDDABLE_SECTIONS:
            cid = row.get(f"{section}_chunk_id")
            if cid:
                chunk_to_resume[cid] = row["resume_id"]
                chunk_to_section[cid] = section
                break  # one chunk per resume — stop at first found

    # Embed query (unit-normalised)
    embedder = GeminiEmbedder(embedding_model)
    query_vector = np.asarray(embedder.embed_query(request.query), dtype=np.float32)
    norm = np.linalg.norm(query_vector)
    if norm > 0:
        query_vector = query_vector / norm

    # Targeted cosine similarity against SQL-filtered chunks only
    vfs = VectorFileStore()
    all_vectors, all_ids = vfs.read(UNIVERSAL_VECTOR_STORE)
    score_by_chunk: dict[str, float] = {}

    if len(all_ids) > 0:
        id_to_pos = {cid: i for i, cid in enumerate(all_ids)}
        target_chunks = [cid for cid in chunk_to_resume if cid in id_to_pos]
        if target_chunks:
            positions = [id_to_pos[cid] for cid in target_chunks]
            subset = all_vectors[positions]
            scores = (subset @ query_vector).tolist()
            for cid, score in zip(target_chunks, scores):
                score_by_chunk[cid] = float(score)

    if not score_by_chunk:
        # Vector store has no vectors for these resumes (embeddings not found).
        # Fall back to returning SQL results in their original order rather than
        # returning empty — the user already has valid RDS matches.
        logger.warning(
            "Vector search found no scoring chunks for %d SQL-matched resumes — "
            "falling back to RDS result order", len(resume_rows)
        )
        seen_users_fb: set[str] = set()
        fallback_candidates: list[CandidateResult] = []
        ordered_fb = [rid for rid in resume_ids if rid in resume_rows_by_id]
        remaining_fb = [r["resume_id"] for r in resume_rows if r["resume_id"] not in set(ordered_fb)]
        for rid in ordered_fb + remaining_fb:
            row = resume_rows_by_id.get(rid, {})
            uid = row.get("user_id", rid)
            if uid in seen_users_fb:
                continue
            seen_users_fb.add(uid)
            fallback_candidates.append(CandidateResult(
                user_id=uid,
                resume_id=rid,
                source_filename=row.get("source_filename", ""),
                similarity_score=None,
                name=row.get("name"),
                email=row.get("email"),
                phone=row.get("phone"),
                location=row.get("location"),
                work_experience_years=row.get("work_experience_years"),
                skills=list(row.get("skills") or []),
                objectives=row.get("objectives"),
                matched_sections=[],
                match_type="rds",
            ))
            if len(fallback_candidates) == k:
                break
        logs = QueryLogs(
            user_query=request.query,
            sql_query=sql,
            sql_matched_count=sql_matched_count,
            routing_decision="rds_and_vector",
            routing_reason=routing_reason + " (vector store empty — showing SQL results)",
            vector_search_used=False,
            vector_query=request.query,
        )
        return RetrieveResponse(query=request.query, k_used=k, candidates=fallback_candidates, logs=logs)

    # Best score + matched sections per resume
    best_score: dict[str, float] = {}
    matched_sections: dict[str, list[str]] = {}
    for cid, score in score_by_chunk.items():
        rid = chunk_to_resume[cid]
        section = chunk_to_section[cid]
        if rid not in best_score or score > best_score[rid]:
            best_score[rid] = score
        matched_sections.setdefault(rid, [])
        if section not in matched_sections[rid]:
            matched_sections[rid].append(section)

    ranked_ids = sorted(best_score, key=lambda r: best_score[r], reverse=True)

    # Deduplicate by user_id
    seen_users_v: set[str] = set()
    deduped_ids: list[str] = []
    for rid in ranked_ids:
        uid = resume_rows_by_id.get(rid, {}).get("user_id") or rid
        if uid not in seen_users_v:
            seen_users_v.add(uid)
            deduped_ids.append(rid)
        if len(deduped_ids) == k:
            break

    candidates_v: list[CandidateResult] = []
    for rid in deduped_ids:
        row = resume_rows_by_id.get(rid, {})
        candidates_v.append(CandidateResult(
            user_id=row.get("user_id", rid),
            resume_id=rid,
            source_filename=row.get("source_filename", ""),
            similarity_score=round(best_score[rid], 6),
            name=row.get("name"),
            email=row.get("email"),
            phone=row.get("phone"),
            location=row.get("location"),
            work_experience_years=row.get("work_experience_years"),
            skills=list(row.get("skills") or []),
            objectives=row.get("objectives"),
            matched_sections=matched_sections.get(rid, []),
            match_type="vector",
        ))

    # Determine which section was actually scored (work_experience_text or projects)
    sections_used = set(chunk_to_section[cid] for cid in score_by_chunk)
    vector_section_used = next(iter(sections_used)) if len(sections_used) == 1 else "mixed"

    logs = QueryLogs(
        user_query=request.query,
        sql_query=sql,
        sql_matched_count=sql_matched_count,
        routing_decision="rds_and_vector",
        routing_reason=routing_reason,
        vector_search_used=True,
        vector_section_used=vector_section_used,
        vector_query=request.query,
    )

    logger.info(
        "RDS + Vector complete: %d unique candidates from %d resumes",
        len(candidates_v), len(resume_rows),
    )
    return RetrieveResponse(query=request.query, k_used=k, candidates=candidates_v, logs=logs)
