from contextlib import asynccontextmanager
import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import text

from drl_wizard.backend.routers import training_route
from drl_wizard.backend.services.storage.database import engine, Base


def _get_cors_origins() -> list[str]:
    raw_origins = os.getenv("DRL_WIZARD_CORS_ORIGINS", "").strip()
    if not raw_origins:
        return ["*"]
    origins = [origin.strip() for origin in raw_origins.split(",") if origin.strip()]
    return origins or ["*"]


def create_app() -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # Create tables once on startup (async DDL)
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
            # Optional: SQLite WAL for better concurrency
            if str(engine.url).startswith("sqlite+aiosqlite://"):
                await conn.execute(text("PRAGMA journal_mode=WAL;"))
                await conn.execute(text("PRAGMA synchronous=NORMAL;"))
        yield
        # optional: await engine.dispose()

    app = FastAPI(lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=_get_cors_origins(),
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["Content-Disposition"],
    )
    # Base.metadata.create_all(bind=engine)
    app.include_router(training_route.router)
    @app.get("/healthz")
    def healthz():
        return {"ok": True}
    return app
app = create_app()
