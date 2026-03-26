from __future__ import annotations

import logging
import uuid

import numpy as np

from app.config import get_config
from app.models import IngestResponse, StrategyResult
from app.services.chunking.registry import get_chunker_registry
from app.services.embedding.embedder import GeminiEmbedder
from app.services.pdf_extractor import extract_pdf_pages
from app.services.store.postgres_store import PostgresStore
from app.services.store.vector_file_store import VectorFileStore
from app.utils import now_ist, now_ist_iso, slugify_name


SUPPORTED_CHUNKING_STRATEGIES = {"fixed_length", "paragraph", "both"}
logger = logging.getLogger("nexvec.ingestion")


def _normalise_vector(vector: list[float]) -> list[float]:
    array = np.asarray(vector, dtype=float)
    norm = np.linalg.norm(array)
    if norm == 0:
        return [0.0 for _ in vector]
    return (array / norm).astype(float).tolist()


UNIVERSAL_VECTOR_STORE = "nex_vec"


def _resolve_kb_name(source_filename: str, provided_kb_name: str | None) -> str:
    if provided_kb_name:
        return slugify_name(provided_kb_name)
    return UNIVERSAL_VECTOR_STORE


def _chunk_page_text(chunker, page_number: int, text: str) -> list[tuple[int, str]]:
    chunks = chunker.split(text)
    return [(page_number, chunk) for chunk in chunks]


def ingest_file(
    file_name: str,
    file_path: str,
    kb_name: str | None,
    chunking_strategy: str,
    chunk_size: int | None,
    overlap_size: int | None,
    embedding_model: str | None,
    overwrite: bool,
) -> IngestResponse:
    config = get_config()
    pg = PostgresStore()
    vfs = VectorFileStore()
    pg.ensure_table()

    resolved_kb_name = _resolve_kb_name(file_name, kb_name)
    logger.info("Resolved kb_name=%s for source=%s", resolved_kb_name, file_name)

    if chunking_strategy not in SUPPORTED_CHUNKING_STRATEGIES:
        raise ValueError("Unsupported chunking strategy requested.")

    strategy_names = ["fixed_length", "paragraph"] if chunking_strategy == "both" else [chunking_strategy]
    active_model = embedding_model or config.embedding_model

    # Duplicate detection
    active_duplicates_by_strategy = {
        strategy: pg.has_active_chunks(file_name, strategy, active_model, resolved_kb_name)
        for strategy in strategy_names
    }

    if not overwrite:
        duplicate_hit = next((s for s, has_dup in active_duplicates_by_strategy.items() if has_dup), None)
        if duplicate_hit:
            logger.info(
                "Duplicate ingestion rejected source=%s kb_name=%s strategy=%s model=%s",
                file_name, resolved_kb_name, duplicate_hit, active_model,
            )
            raise FileExistsError(
                f"Active chunks already exist for file={file_name}, strategy={duplicate_hit}, model={active_model}"
            )

    if overwrite and any(active_duplicates_by_strategy.values()):
        logger.info("Overwrite enabled; deactivating matching rows in kb_name=%s", resolved_kb_name)
        deactivated_at = now_ist_iso()
        deactivated_ids = pg.deactivate_chunks(file_name, strategy_names, active_model, resolved_kb_name, deactivated_at)
        if deactivated_ids:
            vfs.remove_chunk_ids(UNIVERSAL_VECTOR_STORE, set(deactivated_ids))
            logger.info("Removed %s vectors from flat file for kb_name=%s", len(deactivated_ids), resolved_kb_name)

    pages = extract_pdf_pages(file_path)
    if not pages:
        raise LookupError("NO_EXTRACTABLE_TEXT")
    logger.info("Extracted %s text pages from %s", len(pages), file_name)

    effective_chunk_size = chunk_size or config.chunk_size
    effective_overlap_size = overlap_size if overlap_size is not None else config.overlap_size
    if effective_overlap_size >= effective_chunk_size:
        raise ValueError("overlap_size must be smaller than chunk_size")

    chunkers = get_chunker_registry(effective_chunk_size, effective_overlap_size)
    embedder = GeminiEmbedder(active_model)
    created_at = now_ist_iso()

    strategies_processed: list[StrategyResult] = []
    for strategy_name in strategy_names:
        logger.info("Starting chunking strategy=%s", strategy_name)
        chunker = chunkers[strategy_name]
        indexed_chunks: list[tuple[int, str]] = []
        for page in pages:
            indexed_chunks.extend(_chunk_page_text(chunker, page.page_number, page.text))

        chunk_texts = [chunk_text for _, chunk_text in indexed_chunks]
        logger.info("Generating embeddings strategy=%s chunk_count=%s model=%s", strategy_name, len(chunk_texts), embedder.model)
        vectors = embedder.embed_texts(chunk_texts)
        if len(vectors) != len(indexed_chunks):
            raise RuntimeError("Embedding vector count does not match chunk count.")

        chunk_ids: list[str] = []
        pg_rows: list[dict] = []
        normalised_vectors: list[list[float]] = []

        for index, ((page_number, chunk_text), vector) in enumerate(zip(indexed_chunks, vectors)):
            cid = str(uuid.uuid4())
            chunk_ids.append(cid)
            normalised_vectors.append(_normalise_vector(vector))
            pg_rows.append({
                "chunk_id": cid,
                "kb_name": resolved_kb_name,
                "source_filename": file_name,
                "chunking_strategy": strategy_name,
                "chunk_index": index,
                "page_number": page_number,
                "chunk_text": chunk_text,
                "embedding_model": embedder.model,
                "is_active": True,
                "created_at": created_at,
                "deactivated_at": None,
            })

        pg.insert_chunks(pg_rows)
        vfs.append(UNIVERSAL_VECTOR_STORE, chunk_ids, np.array(normalised_vectors, dtype=np.float32))
        logger.info("Stored %s chunks for strategy=%s kb_name=%s", len(chunk_ids), strategy_name, resolved_kb_name)

        strategies_processed.append(
            StrategyResult(
                strategy_name=strategy_name,
                chunk_count=len(chunk_ids),
                embedding_model=embedder.model,
                vector_size=config.vector_size,
                overwritten=overwrite and bool(active_duplicates_by_strategy.get(strategy_name)),
            )
        )

    return IngestResponse(
        kb_name=resolved_kb_name,
        source_filename=file_name,
        strategies_processed=strategies_processed,
        ingested_at=created_at,
    )
