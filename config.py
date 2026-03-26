from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

@dataclass(frozen=True)
class AppConfig:
    document_store_path: Path
    metadata_store_type: str
    db_host: str
    db_port: str
    db_name: str
    db_user: str
    db_password: str

_CONFIG_CACHE: AppConfig | None = None

def project_root() -> Path:
    return Path(__file__).resolve().parents[1]

def get_config() -> AppConfig:
    global _CONFIG_CACHE
    if _CONFIG_CACHE is not None:
        return _CONFIG_CACHE
    
    raw_path = os.getenv("DOCUMENT_STORE_PATH", "./document_store/")
    document_store_path = Path(raw_path)
    if not document_store_path.is_absolute():
        document_store_path = (project_root() / document_store_path).resolve()

    _CONFIG_CACHE = AppConfig(
        document_store_path=document_store_path,
        metadata_store_type=os.getenv("METADATA_STORE_TYPE", "jsonl"),
        db_host=os.getenv("DB_HOST", "localhost"),
        db_port=os.getenv("DB_PORT", "5432"),
        db_name=os.getenv("DB_NAME", "postgres"),
        db_user=os.getenv("DB_USER", "postgres"),
        db_password=os.getenv("DB_PASSWORD", ""),
    )
    return _CONFIG_CACHE
