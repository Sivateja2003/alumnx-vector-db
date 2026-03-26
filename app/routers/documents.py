from __future__ import annotations

from fastapi import APIRouter, HTTPException

from app.services.store.postgres_store import PostgresStore
from app.services.store.vector_file_store import VectorFileStore
from app.services.ingestion import UNIVERSAL_VECTOR_STORE


router = APIRouter()


@router.get("/documents/{filename}")
def get_document(filename: str) -> dict:
    pg = PostgresStore()
    pg.ensure_table()
    doc = pg.get_document(filename)
    if not doc:
        raise HTTPException(status_code=404, detail={"error": "DOCUMENT_NOT_FOUND", "message": f"No active document found with filename '{filename}'."})
    return doc


@router.get("/documents")
def list_documents() -> dict:
    pg = PostgresStore()
    pg.ensure_table()
    return {"documents": pg.list_documents()}


@router.delete("/documents/{filename}")
def delete_document(filename: str) -> dict:
    pg = PostgresStore()
    vfs = VectorFileStore()
    pg.ensure_table()

    deactivated_ids = pg.delete_document(filename)
    if not deactivated_ids:
        raise HTTPException(status_code=404, detail={"error": "DOCUMENT_NOT_FOUND", "message": f"No active document found with filename '{filename}'."})

    vfs.remove_chunk_ids(UNIVERSAL_VECTOR_STORE, set(deactivated_ids))
    return {"deleted": filename, "chunks_removed": len(deactivated_ids)}
