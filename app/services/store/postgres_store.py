from __future__ import annotations

import psycopg2
import psycopg2.extras

from app.config import get_config

_CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS chunks (
    chunk_id          VARCHAR(36)  PRIMARY KEY,
    kb_name           VARCHAR(255) NOT NULL,
    source_filename   VARCHAR(500) NOT NULL,
    chunking_strategy VARCHAR(50)  NOT NULL,
    chunk_index       INTEGER      NOT NULL,
    page_number       INTEGER,
    chunk_text        TEXT         NOT NULL,
    embedding_model   VARCHAR(255) NOT NULL,
    is_active         BOOLEAN      NOT NULL DEFAULT TRUE,
    created_at        TIMESTAMPTZ  NOT NULL,
    deactivated_at    TIMESTAMPTZ
);
CREATE INDEX IF NOT EXISTS idx_chunks_kb_name ON chunks(kb_name);
CREATE INDEX IF NOT EXISTS idx_chunks_active  ON chunks(is_active);
CREATE INDEX IF NOT EXISTS idx_chunks_source  ON chunks(source_filename, chunking_strategy, embedding_model);
"""


class PostgresStore:
    """Stores all chunk metadata (no vectors) in PostgreSQL."""

    def __init__(self) -> None:
        self.config = get_config()

    def _connect(self):
        return psycopg2.connect(self.config.postgres_url)

    def ensure_table(self) -> None:
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(_CREATE_TABLE_SQL)
            conn.commit()

    def insert_chunks(self, rows: list[dict]) -> None:
        if not rows:
            return
        sql = """
            INSERT INTO chunks (
                chunk_id, kb_name, source_filename, chunking_strategy,
                chunk_index, page_number, chunk_text, embedding_model,
                is_active, created_at, deactivated_at
            ) VALUES (
                %(chunk_id)s, %(kb_name)s, %(source_filename)s, %(chunking_strategy)s,
                %(chunk_index)s, %(page_number)s, %(chunk_text)s, %(embedding_model)s,
                %(is_active)s, %(created_at)s, %(deactivated_at)s
            )
        """
        with self._connect() as conn:
            with conn.cursor() as cur:
                psycopg2.extras.execute_batch(cur, sql, rows)
            conn.commit()

    def has_active_chunks(self, source_filename: str, chunking_strategy: str, embedding_model: str, kb_name: str) -> bool:
        sql = """
            SELECT 1 FROM chunks
            WHERE source_filename = %s
              AND chunking_strategy = %s
              AND embedding_model = %s
              AND kb_name = %s
              AND is_active = TRUE
            LIMIT 1
        """
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (source_filename, chunking_strategy, embedding_model, kb_name))
                return cur.fetchone() is not None

    def deactivate_chunks(
        self,
        source_filename: str,
        strategy_names: list[str],
        embedding_model: str,
        kb_name: str,
        deactivated_at: str,
    ) -> list[str]:
        """Set is_active=False for matching chunks. Returns the deactivated chunk_ids."""
        sql = """
            UPDATE chunks
            SET is_active = FALSE, deactivated_at = %s
            WHERE source_filename = %s
              AND chunking_strategy = ANY(%s)
              AND embedding_model = %s
              AND kb_name = %s
              AND is_active = TRUE
            RETURNING chunk_id
        """
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (deactivated_at, source_filename, strategy_names, embedding_model, kb_name))
                rows = cur.fetchall()
            conn.commit()
        return [row[0] for row in rows]

    def get_active_group_index(self, kb_name: str, embedding_model: str | None = None) -> list[dict]:
        """Return [{chunk_id, chunking_strategy, embedding_model}] for all active chunks in a KB."""
        if embedding_model:
            sql = """
                SELECT chunk_id, chunking_strategy, embedding_model
                FROM chunks
                WHERE kb_name = %s AND embedding_model = %s AND is_active = TRUE
            """
            params = (kb_name, embedding_model)
        else:
            sql = """
                SELECT chunk_id, chunking_strategy, embedding_model
                FROM chunks
                WHERE kb_name = %s AND is_active = TRUE
            """
            params = (kb_name,)

        with self._connect() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(sql, params)
                return [dict(row) for row in cur.fetchall()]

    def get_chunks_by_ids(self, chunk_ids: list[str]) -> list[dict]:
        """Fetch full metadata for a list of chunk_ids (used after KNN to enrich results)."""
        if not chunk_ids:
            return []
        sql = """
            SELECT chunk_id, chunk_text, source_filename, chunking_strategy,
                   chunk_index, page_number, embedding_model, created_at
            FROM chunks
            WHERE chunk_id = ANY(%s)
        """
        with self._connect() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(sql, (chunk_ids,))
                return [dict(row) for row in cur.fetchall()]

    def kb_exists(self, kb_name: str) -> bool:
        sql = "SELECT 1 FROM chunks WHERE kb_name = %s LIMIT 1"
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (kb_name,))
                return cur.fetchone() is not None

    def list_kb_names(self) -> list[str]:
        sql = "SELECT DISTINCT kb_name FROM chunks WHERE is_active = TRUE ORDER BY kb_name"
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(sql)
                return [row[0] for row in cur.fetchall()]

    def get_document(self, source_filename: str) -> dict | None:
        sql = """
            SELECT source_filename, kb_name, MIN(created_at) AS uploaded_at,
                   COUNT(*) AS chunk_count,
                   array_agg(DISTINCT chunking_strategy) AS chunking_strategies,
                   array_agg(DISTINCT embedding_model) AS embedding_models
            FROM chunks
            WHERE source_filename = %s AND is_active = TRUE
            GROUP BY source_filename, kb_name
        """
        with self._connect() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(sql, (source_filename,))
                row = cur.fetchone()
                if not row:
                    return None
                return {
                    "source_filename": row["source_filename"],
                    "kb_name": row["kb_name"],
                    "uploaded_at": str(row["uploaded_at"]),
                    "chunk_count": row["chunk_count"],
                    "chunking_strategies": list(row["chunking_strategies"]),
                    "embedding_models": list(row["embedding_models"]),
                }

    def list_documents(self) -> list[dict]:
        sql = """
            SELECT source_filename, kb_name, MIN(created_at) AS uploaded_at, COUNT(*) AS chunk_count
            FROM chunks
            WHERE is_active = TRUE
            GROUP BY source_filename, kb_name
            ORDER BY MIN(created_at) DESC
        """
        with self._connect() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(sql)
                return [
                    {
                        "source_filename": row["source_filename"],
                        "kb_name": row["kb_name"],
                        "uploaded_at": str(row["uploaded_at"]),
                        "chunk_count": row["chunk_count"],
                    }
                    for row in cur.fetchall()
                ]

    def delete_document(self, source_filename: str) -> list[str]:
        """Deactivate all active chunks for a document. Returns deactivated chunk_ids."""
        sql = """
            UPDATE chunks
            SET is_active = FALSE, deactivated_at = NOW()
            WHERE source_filename = %s AND is_active = TRUE
            RETURNING chunk_id
        """
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (source_filename,))
                rows = cur.fetchall()
            conn.commit()
        return [row[0] for row in rows]
