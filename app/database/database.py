"""SQLAlchemy engine + сессии для SQLite (data/app.db)."""
import logging
from pathlib import Path

from sqlalchemy import create_engine, event
from sqlalchemy.orm import DeclarativeBase, sessionmaker

from app.config import BASE_DIR, settings

logger = logging.getLogger("app.database")


class Base(DeclarativeBase):
    pass


def _resolve_url(url: str) -> str:
    """Относительный sqlite-путь приводим к абсолютному от корня проекта,
    чтобы приложение находило базу при запуске из любой директории."""
    prefix = "sqlite:///./"
    if url.startswith(prefix):
        path = (BASE_DIR / url[len(prefix):]).resolve()
        return f"sqlite:///{path.as_posix()}"
    return url


RESOLVED_DB_URL = _resolve_url(settings.database_url)
_IS_SQLITE = RESOLVED_DB_URL.startswith("sqlite")

if _IS_SQLITE:
    db_file = Path(RESOLVED_DB_URL[len("sqlite:///"):])
    db_file.parent.mkdir(parents=True, exist_ok=True)

engine = create_engine(
    RESOLVED_DB_URL,
    connect_args={"check_same_thread": False, "timeout": 30} if _IS_SQLITE else {},
)

if _IS_SQLITE:
    @event.listens_for(engine, "connect")
    def _sqlite_pragma(dbapi_connection, _record):
        # WAL + busy timeout: параллельные записи из AI-потока и API-хендлеров
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA journal_mode=WAL")
        cursor.execute("PRAGMA synchronous=NORMAL")
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()

SessionLocal = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)


def init_db() -> None:
    from app.database import models  # noqa: F401 — регистрация таблиц

    Base.metadata.create_all(engine)
    logger.info("База данных готова: %s", RESOLVED_DB_URL)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
