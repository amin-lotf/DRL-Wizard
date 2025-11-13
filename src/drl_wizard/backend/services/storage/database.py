# backend/services/storage/database.py
from __future__ import annotations

from typing import Optional, AsyncIterator
from datetime import datetime

from sqlalchemy import (
    String, DateTime, Boolean, ForeignKey, Enum, Integer, UniqueConstraint, and_, text
)
from sqlalchemy.ext.asyncio import (
    create_async_engine, AsyncSession, async_sessionmaker
)
from sqlalchemy.orm import (
    Mapped, relationship, mapped_column, DeclarativeBase
)
from drl_wizard.backend.settings import SQLALCHEMY_DATABASE_URI
from drl_wizard.common.types import JobStatus, ResultType, AlgoType


# --- Helpers -----------------------------------------------------------------

def _to_async_sqlite_dsn(dsn: str) -> str:
    """
    If user provided a sync SQLite DSN (e.g., 'sqlite:///app.db'),
    convert it to the async driver DSN ('sqlite+aiosqlite:///app.db').
    Otherwise, return as-is.
    """
    if dsn.startswith("sqlite:///") or dsn.startswith("sqlite:///:"):
        return "sqlite+aiosqlite" + dsn[len("sqlite"):]
    if dsn.startswith("sqlite://"):
        # rare forms; still convert
        return "sqlite+aiosqlite" + dsn[len("sqlite"):]
    return dsn


ASYNC_DATABASE_URI = _to_async_sqlite_dsn(SQLALCHEMY_DATABASE_URI)

# For SQLite + aiosqlite, don't pass check_same_thread; it's a sync-driver option.
engine = create_async_engine(
    ASYNC_DATABASE_URI,
    echo=False,                # turn True if you want verbose SQL logs
    pool_pre_ping=True,        # keep connections healthy
    future=True,
)

# Async session factory
SessionLocal = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    autoflush=False,
    expire_on_commit=False,
)

# --- Declarative Base --------------------------------------------------------

class Base(DeclarativeBase):
    pass


# --- Models ------------------------------------------------------------------

class JobResultsModel(Base):
    __tablename__ = "job_train_results"

    id: Mapped[int] = mapped_column(primary_key=True, index=True)
    job_id: Mapped[int] = mapped_column(ForeignKey("jobs.job_id", ondelete="CASCADE"), index=True)

    result_type: Mapped[ResultType] = mapped_column(
        Enum(ResultType, name="result_type", native_enum=False, validate_string=True),
        nullable=False,
    )
    manifest_uri: Mapped[str] = mapped_column(String, nullable=False)
    segment_steps: Mapped[int] = mapped_column(Integer, nullable=False)
    latest_step: Mapped[int] = mapped_column(Integer, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)

    job: Mapped["JobModel"] = relationship("JobModel", back_populates="results")

    __table_args__ = (
        UniqueConstraint("job_id", "result_type", name="uq_job_result_per_type"),
    )


class JobModel(Base):
    __tablename__ = "jobs"

    job_id: Mapped[int] = mapped_column(primary_key=True, index=True)
    status: Mapped[JobStatus] = mapped_column(
        Enum(JobStatus, name="job_status", native_enum=False, validate_string=True),
        default=JobStatus.QUEUED,
        nullable=False,
    )
    env_id: Mapped[str] = mapped_column(String, nullable=False)
    algo_id: Mapped[AlgoType] = mapped_column(
        Enum(AlgoType, name="algo_type", native_enum=False, validate_string=True),
        nullable=False,
    )

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    started_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    finished_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    detail: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    stop_requested: Mapped[bool] = mapped_column(Boolean, default=False)

    results: Mapped[list["JobResultsModel"]] = relationship(
        "JobResultsModel",
        cascade="all,delete-orphan",
        back_populates="job",
        lazy="selectin",
    )

    evaluate_results: Mapped[Optional["JobResultsModel"]] = relationship(
        "JobResultsModel",
        primaryjoin=lambda: and_(
            JobResultsModel.job_id == JobModel.job_id,
            JobResultsModel.result_type == ResultType.EVALUATE,
        ),
        uselist=False,
        viewonly=True,
        lazy="selectin",
    )

    train_results: Mapped[Optional["JobResultsModel"]] = relationship(
        "JobResultsModel",
        primaryjoin=lambda: and_(
            JobResultsModel.job_id == JobModel.job_id,
            JobResultsModel.result_type == ResultType.TRAIN,
        ),
        uselist=False,
        viewonly=True,
        lazy="selectin",
    )


# --- FastAPI dependency ------------------------------------------------------

async def get_db() -> AsyncIterator[AsyncSession]:
    """
    Async DB session dependency for FastAPI.
    """
    async with SessionLocal() as session:
        try:
            yield session
        finally:
            # 'async with' handles close; explicit close() is not required.
            pass


# --- (Optional) Startup table creation & WAL --------------------------------
# Put this in your FastAPI app startup (lifespan) once, not on every import.
# Example is included here for convenience; call it from app.lifespan/startup.

async def init_db() -> None:
    # Create tables
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

        # If SQLite, enable WAL for better concurrent readers/writers
        if ASYNC_DATABASE_URI.startswith("sqlite+aiosqlite://"):
            await conn.execute(text("PRAGMA journal_mode=WAL;"))
            await conn.execute(text("PRAGMA synchronous=NORMAL;"))
