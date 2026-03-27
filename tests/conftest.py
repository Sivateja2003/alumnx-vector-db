from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest
from dotenv import load_dotenv


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    load_dotenv(ROOT / ".env")
except PermissionError:
    pass  # Fallback to process environment or skip markers if API key missing

# Ensure config-level env vars are present for tests (services are mocked, no real DB connection is made)
os.environ.setdefault("POSTGRES_URL", "postgresql://test:test@localhost/test")



def pytest_collection_modifyitems(config, items):
    if not os.environ.get("GOOGLE_API_KEY"):
        skip_marker = pytest.mark.skip(reason="GOOGLE_API_KEY is not set; skipping Alumnx Vector DB test suite.")
        for item in items:
            item.add_marker(skip_marker)
