# tests/conftest.py
import os
import sys
from pathlib import Path

# Put the project root (the folder that contains "backend/") on sys.path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Make sure settings-dependent imports don't fail during import-time engine creation
# Your database.py already adapts sqlite:/// -> sqlite+aiosqlite://
os.environ.setdefault("SQLALCHEMY_DATABASE_URI", "sqlite:///./test.db")
