import os
from pathlib import Path

DATABASE_PATH = Path(os.getenv("DATABASE_PATH", "./db/drl.db"))
DATABASE_PATH.parent.mkdir(parents=True, exist_ok=True)

SQLALCHEMY_DATABASE_URI = f"sqlite:///{DATABASE_PATH}"