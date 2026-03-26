from __future__ import annotations

from fastapi import APIRouter

from app.services.store.postgres_store import PostgresStore


router = APIRouter()


@router.get("/knowledgebases")
def list_knowledgebases() -> dict:
    pg = PostgresStore()
    pg.ensure_table()
    return {"knowledgebases": pg.list_kb_names()}

