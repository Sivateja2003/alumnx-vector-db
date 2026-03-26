from __future__ import annotations

import logging
from collections import defaultdict

import numpy as np

from app.config import get_config
from app.models import KBResult, ChunkResult, RetrieveRequest, RetrieveResponse, RetrievalStrategy, StrategyGroupResult
from app.services.embedding.embedder import GeminiEmbedder
from app.services.retrieval.registry import get_retriever_registry
from app.services.store.postgres_store import PostgresStore
from app.services.store.vector_file_store import VectorFileStore
from app.services.ingestion import UNIVERSAL_VECTOR_STORE
from app.utils import slugify_name


SUPPORTED_ALGORITHMS = {"knn"}
SUPPORTED_DISTANCE_METRICS = {"cosine", "dot_product"}
logger = logging.getLogger("nexvec.retrieval")


def _default_retrieval_strategy() -> RetrievalStrategy:
    config = get_config()
    return RetrievalStrategy(
        algorithm=config.default_retrieval_strategy.get("algorithm", "knn"),
        distance_metric=config.default_retrieval_strategy.get("distance_metric", "cosine"),
    )


def _validate_retrieval_strategy(strategy: RetrievalStrategy) -> RetrievalStrategy:
    if strategy.algorithm == "ann":
        raise ValueError("WARNING: Retrieval algorithm 'ann' is not supported in Phase 1.")
    if strategy.algorithm not in SUPPORTED_ALGORITHMS:
        raise ValueError(f"Unsupported retrieval algorithm: {strategy.algorithm}")
    if strategy.distance_metric not in SUPPORTED_DISTANCE_METRICS:
        raise ValueError(f"Unsupported distance metric: {strategy.distance_metric}")
    return strategy


def _normalize(vector: list[float]) -> list[float]:
    array = np.asarray(vector, dtype=float)
    norm = np.linalg.norm(array)
    if norm == 0:
        return [0.0 for _ in vector]
    return (array / norm).astype(float).tolist()


def _retrieve_for_kb(
    kb_name: str,
    query: str,
    k: int,
    strategy: RetrievalStrategy,
    embedding_model: str | None,
    pg: PostgresStore,
    vfs: VectorFileStore,
) -> list[StrategyGroupResult]:
    # Load all active chunk_id → (chunking_strategy, embedding_model) from PostgreSQL
    group_index = pg.get_active_group_index(kb_name, embedding_model)
    if not group_index:
        logger.info("No active chunks found in kb_name=%s", kb_name)
        return []

    # Load vectors from the universal flat file
    all_vectors, all_chunk_ids = vfs.read(UNIVERSAL_VECTOR_STORE)
    if len(all_chunk_ids) == 0:
        return []

    chunk_id_to_pos = {cid: i for i, cid in enumerate(all_chunk_ids)}

    # Group active chunks by (chunking_strategy, embedding_model)
    groups: dict[tuple[str, str], list[tuple[str, int]]] = defaultdict(list)
    for entry in group_index:
        cid = entry["chunk_id"]
        if cid in chunk_id_to_pos:
            key = (entry["chunking_strategy"], entry["embedding_model"])
            groups[key].append((cid, chunk_id_to_pos[cid]))

    results: list[StrategyGroupResult] = []
    for (chunking_strategy, model), id_pos_pairs in sorted(groups.items()):
        group_ids = [cid for cid, _ in id_pos_pairs]
        positions = [pos for _, pos in id_pos_pairs]
        group_vectors = all_vectors[positions]

        embedder = GeminiEmbedder(model)
        logger.info(
            "Retrieving group chunking_strategy=%s model=%s row_count=%s k=%s metric=%s",
            chunking_strategy, model, len(group_ids), k, strategy.distance_metric,
        )
        query_vector = np.asarray(embedder.embed_query(query), dtype=np.float32)
        if strategy.distance_metric == "cosine":
            norm = np.linalg.norm(query_vector)
            if norm > 0:
                query_vector = query_vector / norm

        # Pass numpy matrix directly — no Python list round-trip
        retriever = get_retriever_registry()["knn"]
        top_results = retriever.retrieve(
            query_vector=query_vector,
            vectors=group_vectors,
            chunk_ids=group_ids,
            k=k,
            distance_metric=strategy.distance_metric,
        )
        top_chunk_ids = [cid for cid, _ in top_results]
        similarity_by_id = {cid: score for cid, score in top_results}

        # O(1) vector lookup by chunk_id — no list.index() scan
        id_to_pos = {cid: i for i, cid in enumerate(group_ids)}

        # Fetch full metadata for top-k from PostgreSQL
        metadata_rows = pg.get_chunks_by_ids(top_chunk_ids)
        meta_by_id = {r["chunk_id"]: r for r in metadata_rows}

        chunks = [
            ChunkResult(
                chunk_id=cid,
                similarity_score=similarity_by_id[cid],
                chunk_text=meta_by_id[cid]["chunk_text"],
                embedding_vector=group_vectors[id_to_pos[cid]].tolist(),
                source_filename=meta_by_id[cid]["source_filename"],
                chunk_index=meta_by_id[cid]["chunk_index"],
                page_number=meta_by_id[cid].get("page_number"),
                created_at=str(meta_by_id[cid]["created_at"]),
            )
            for cid in top_chunk_ids
            if cid in meta_by_id
        ]

        results.append(StrategyGroupResult(
            chunking_strategy=chunking_strategy,
            embedding_model=model,
            chunks=chunks,
        ))

    return results


def retrieve_documents(request: RetrieveRequest) -> RetrieveResponse:
    config = get_config()
    strategy = request.retrieval_strategy or _default_retrieval_strategy()
    strategy = _validate_retrieval_strategy(strategy)
    k = request.k or config.knn_k
    pg = PostgresStore()
    vfs = VectorFileStore()
    pg.ensure_table()

    logger.info("Retrieve pipeline started query=%r kb_name=%s k=%s embedding_model=%s", request.query, request.kb_name, k, request.embedding_model)

    if not request.query.strip():
        raise ValueError("EMPTY_QUERY")

    if request.kb_name:
        resolved = slugify_name(request.kb_name)
        if not pg.kb_exists(resolved):
            raise FileNotFoundError(resolved)
        kb_names = [resolved]
    else:
        kb_names = pg.list_kb_names()
        if not kb_names:
            logger.info("No knowledge bases found")
            return RetrieveResponse(query=request.query, retrieval_strategy_used=strategy, k_used=k, results=[])

    kb_results: list[KBResult] = []
    for kb_name in kb_names:
        strategy_results = _retrieve_for_kb(kb_name, request.query, k, strategy, request.embedding_model, pg, vfs)
        kb_results.append(KBResult(kb_name=kb_name, strategy_results=strategy_results))

    logger.info("Retrieve pipeline completed kb_count=%s", len(kb_results))
    return RetrieveResponse(
        query=request.query,
        retrieval_strategy_used=strategy,
        k_used=k,
        results=kb_results,
    )
