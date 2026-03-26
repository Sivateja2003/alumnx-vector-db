import os
import json
import psycopg2
import psycopg2.extras
from typing import Optional
from app.config import get_config

class BaseDocumentRegistry:
    def read_all(self) -> list[dict]: raise NotImplementedError
    def add_record(self, record: dict): raise NotImplementedError
    def get_record(self, doc_id: str) -> Optional[dict]: raise NotImplementedError
    def delete_record(self, doc_id: str) -> bool: raise NotImplementedError
    def find_by_hash(self, file_hash: str) -> Optional[dict]: raise NotImplementedError
    def close(self): pass

class JSONLDocumentRegistry(BaseDocumentRegistry):
    def __init__(self):
        self.registry_file = get_config().document_store_path / "registry.jsonl"
        self.registry_file.parent.mkdir(parents=True, exist_ok=True)
        if not self.registry_file.exists():
            self.registry_file.touch()

    def read_all(self) -> list[dict]:
        with self.registry_file.open("r", encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]

    def write_all(self, records: list[dict]):
        with self.registry_file.open("w", encoding="utf-8") as f:
            for record in records:
                f.write(json.dumps(record) + "\n")

    def add_record(self, record: dict):
        records = self.read_all()
        records.append(record)
        self.write_all(records)

    def get_record(self, doc_id: str) -> Optional[dict]:
        records = self.read_all()
        for r in records:
            if r["id"] == doc_id:
                return r
        return None

    def delete_record(self, doc_id: str) -> bool:
        records = self.read_all()
        new_records = [r for r in records if r["id"] != doc_id]
        if len(new_records) == len(records):
            return False
        self.write_all(new_records)
        return True

    def find_by_hash(self, file_hash: str) -> Optional[dict]:
        records = self.read_all()
        for r in records:
            if r["file_hash"] == file_hash:
                return r
        return None

class PostgresDocumentRegistry(BaseDocumentRegistry):
    def __init__(self):
        conf = get_config()
        self.conn = psycopg2.connect(
            host=conf.db_host,
            database=conf.db_name,
            user=conf.db_user,
            password=conf.db_password,
            port=conf.db_port
        )
        self.conn.autocommit = True
        self._init_db()

    def _init_db(self):
        with self.conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS documents (
                    id VARCHAR(36) PRIMARY KEY,
                    file_hash VARCHAR(64) UNIQUE NOT NULL,
                    original_filename VARCHAR(255) NOT NULL,
                    title VARCHAR(255) NOT NULL,
                    description TEXT,
                    kb_name VARCHAR(100),
                    status VARCHAR(50) NOT NULL,
                    file_size_bytes BIGINT NOT NULL,
                    created_at VARCHAR(50) NOT NULL
                )
            """)

    def _serialize_row(self, row) -> dict:
        return dict(row)

    def read_all(self) -> list[dict]:
        with self.conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
            cur.execute("SELECT * FROM documents ORDER BY created_at DESC")
            rows = cur.fetchall()
            return [self._serialize_row(row) for row in rows]

    def add_record(self, record: dict):
        with self.conn.cursor() as cur:
            cur.execute("""
                INSERT INTO documents (
                    id, file_hash, original_filename, title, description,
                    kb_name, status, file_size_bytes, created_at
                ) VALUES (
                    %(id)s, %(file_hash)s, %(original_filename)s, %(title)s, %(description)s,
                    %(kb_name)s, %(status)s, %(file_size_bytes)s, %(created_at)s
                )
            """, record)

    def get_record(self, doc_id: str) -> Optional[dict]:
        with self.conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
            cur.execute("SELECT * FROM documents WHERE id = %s", (doc_id,))
            row = cur.fetchone()
            return self._serialize_row(row) if row else None

    def delete_record(self, doc_id: str) -> bool:
        with self.conn.cursor() as cur:
            cur.execute("DELETE FROM documents WHERE id = %s", (doc_id,))
            return cur.rowcount > 0

    def find_by_hash(self, file_hash: str) -> Optional[dict]:
        with self.conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
            cur.execute("SELECT * FROM documents WHERE file_hash = %s", (file_hash,))
            row = cur.fetchone()
            return self._serialize_row(row) if row else None
            
    def close(self):
        if self.conn:
            self.conn.close()

def get_document_registry():
    store_type = get_config().metadata_store_type.lower()
    if store_type == "postgres":
        registry = PostgresDocumentRegistry()
    else:
        registry = JSONLDocumentRegistry()
        
    try:
        yield registry
    finally:
        registry.close()
