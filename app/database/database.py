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
    _migrate_sqlite()
    logger.info("База данных готова: %s", RESOLVED_DB_URL)


def _migrate_sqlite() -> None:
    """Маленькие миграции существующих таблиц (SQLite ALTER TABLE):
    create_all не добавляет колонки в уже созданные таблицы."""
    if not _IS_SQLITE:
        return
    try:
        with engine.begin() as conn:
            # unknown_events.global_id — трекинг посторонних (межкамерный слой)
            columns = [row[1] for row in
                       conn.exec_driver_sql("PRAGMA table_info(unknown_events)")]
            if columns and "global_id" not in columns:
                conn.exec_driver_sql(
                    "ALTER TABLE unknown_events ADD COLUMN global_id INTEGER")
                logger.info("Миграция: unknown_events + global_id")
    except Exception:
        logger.exception("Миграция SQLite не выполнена")


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
