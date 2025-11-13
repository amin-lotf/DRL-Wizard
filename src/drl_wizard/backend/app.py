from contextlib import asynccontextmanager
from sqlalchemy import text
from fastapi import FastAPI
from drl_wizard.backend.routers import training_route
from drl_wizard.backend.services.storage.database import engine, Base


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
    # Base.metadata.create_all(bind=engine)
    app.include_router(training_route.router)
    @app.get("/healthz")
    def healthz():
        return {"ok": True}
    return app
app = create_app()