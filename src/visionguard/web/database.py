from __future__ import annotations

from datetime import UTC, datetime

from sqlalchemy import DateTime, Float, Integer, String, create_engine, select
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, sessionmaker


class Base(DeclarativeBase):
    pass


class PredictionRecord(Base):
    __tablename__ = "prediction_history"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC), index=True
    )
    original_filename: Mapped[str] = mapped_column(String(255))
    image_sha256: Mapped[str] = mapped_column(String(64), index=True)
    verdict: Mapped[str] = mapped_column(String(32), index=True)
    confidence: Mapped[float] = mapped_column(Float)
    ai_probability: Mapped[float] = mapped_column(Float)
    threshold: Mapped[float] = mapped_column(Float)
    model_version: Mapped[str] = mapped_column(String(128))
    inference_source: Mapped[str] = mapped_column(String(16))


class HistoryStore:
    def __init__(self, database_url: str) -> None:
        connect_args = {"check_same_thread": False} if database_url.startswith("sqlite") else {}
        self.engine = create_engine(database_url, pool_pre_ping=True, connect_args=connect_args)
        self.session_factory = sessionmaker(self.engine, expire_on_commit=False)

    def initialize(self) -> None:
        Base.metadata.create_all(self.engine)

    def add(self, **values) -> PredictionRecord:
        with self.session_factory() as session:
            record = PredictionRecord(**values)
            session.add(record)
            session.commit()
            return record

    def recent(self, limit: int = 100) -> list[PredictionRecord]:
        with self.session_factory() as session:
            statement = (
                select(PredictionRecord).order_by(PredictionRecord.created_at.desc()).limit(limit)
            )
            return list(session.scalars(statement))

    def healthcheck(self) -> bool:
        try:
            with Session(self.engine) as session:
                session.execute(select(1))
            return True
        except Exception:
            return False
