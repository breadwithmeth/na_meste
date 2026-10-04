"""Модели данных.

RTSP URL в базе НЕ хранится — генерируется из параметров.
AI-таблицы (сотрудники, лица, присутствие) хранят эмбеддинги в BLOB.
"""
from datetime import datetime, timezone

from sqlalchemy import Boolean, DateTime, Float, ForeignKey, Index, Integer, LargeBinary, String
from sqlalchemy.ext.hybrid import hybrid_property
from sqlalchemy.orm import Mapped, mapped_column

from app.database.database import Base


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Camera(Base):
    __tablename__ = "cameras"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(100))
    nvr_host: Mapped[str] = mapped_column(String(255))
    rtsp_port: Mapped[int] = mapped_column(Integer, default=554)
    username: Mapped[str] = mapped_column(String(100))
    # Пароль хранится в БД (не в .env), но никогда не отдаётся наружу через API
    password: Mapped[str] = mapped_column(String(255))
    channel: Mapped[int] = mapped_column(Integer)
    stream_type: Mapped[str] = mapped_column(String(10), default="main")  # 'main' | 'sub'
    enabled: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow, onupdate=_utcnow)

    @hybrid_property
    def has_password(self) -> bool:
        return bool(self.password)

    def __repr__(self) -> str:  # pragma: no cover
        return f"<Camera id={self.id} name={self.name!r} host={self.nvr_host} ch={self.channel}>"


class Employee(Base):
    __tablename__ = "employees"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(150))
    external_id: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    active: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow, onupdate=_utcnow)

    def __repr__(self) -> str:  # pragma: no cover
        return f"<Employee id={self.id} name={self.name!r}>"


class EmployeeFace(Base):
    """Эмбеддинг лица сотрудника (ArcFace, 512-d float32 → BLOB).

    Один сотрудник может иметь несколько эмбеддингов (разные углы, очки,
    освещение). thumbnail — маленький JPEG 112x112 выровненного лица
    для управления в UI; исходные фотографии не хранятся.
    """
    __tablename__ = "employee_faces"

    id: Mapped[int] = mapped_column(primary_key=True)
    employee_id: Mapped[int] = mapped_column(
        ForeignKey("employees.id", ondelete="CASCADE"), index=True
    )
    embedding: Mapped[bytes] = mapped_column(LargeBinary)  # float32 × 512
    thumbnail: Mapped[bytes | None] = mapped_column(LargeBinary, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)


class PresenceSession(Base):
    """Сессия присутствия сотрудника на камере.

    Вместо тысяч событий на каждый кадр — одна открытая сессия:
    started_at при первом подтверждении, last_seen_at обновляется,
    ended_at = last_seen_at при исчезновении (по таймауту).
    """
    __tablename__ = "presence_sessions"
    __table_args__ = (
        Index("ix_presence_emp_cam_end", "employee_id", "camera_id", "ended_at"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    employee_id: Mapped[int] = mapped_column(
        ForeignKey("employees.id", ondelete="CASCADE"), index=True
    )
    camera_id: Mapped[int] = mapped_column(
        ForeignKey("cameras.id", ondelete="CASCADE"), index=True
    )
    started_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)
    last_seen_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)
    ended_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    confidence: Mapped[float | None] = mapped_column(Float, nullable=True)


class UnknownEvent(Base):
    """Фиксация постороннего (неопознанного) человека: снимок + камера + время.

    Снимок — JPEG-кроп человека из кадра (доказательство для просмотра в UI
    и отправки в Telegram). global_id связывает фиксацию с глобальной
    личностью (межкамерный трекинг посторонних)."""
    __tablename__ = "unknown_events"

    id: Mapped[int] = mapped_column(primary_key=True)
    camera_id: Mapped[int] = mapped_column(
        ForeignKey("cameras.id", ondelete="CASCADE"), index=True
    )
    track_id: Mapped[int] = mapped_column(Integer)
    global_id: Mapped[int | None] = mapped_column(Integer, nullable=True, index=True)
    snapshot: Mapped[bytes | None] = mapped_column(LargeBinary, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)


class GlobalPerson(Base):
    """Глобальная личность человека (global_id) — единая для всех камер.

    Локальные track_id привязываются к global_id через GlobalIdentityManager
    (composite matching: Re-ID + время + топология + лицо).
    """
    __tablename__ = "global_persons"

    id: Mapped[int] = mapped_column(primary_key=True)          # = global_id
    employee_id: Mapped[int | None] = mapped_column(
        ForeignKey("employees.id", ondelete="SET NULL"), nullable=True
    )
    status: Mapped[str] = mapped_column(String(20), default="ACTIVE")  # NEW/ACTIVE/LOST
    last_camera_id: Mapped[int | None] = mapped_column(
        ForeignKey("cameras.id", ondelete="SET NULL"), nullable=True
    )
    last_track_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)
    last_seen_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)


class GlobalObservation(Base):
    """Наблюдение глобальной личности: появление на камере с новым локальным
    треком. Хранит appearance-эмбеддинг (OSNet, 512-d) — это persistent
    identity gallery — и снимок для таймлайна в UI."""
    __tablename__ = "global_observations"

    id: Mapped[int] = mapped_column(primary_key=True)
    global_id: Mapped[int] = mapped_column(
        ForeignKey("global_persons.id", ondelete="CASCADE"), index=True
    )
    camera_id: Mapped[int] = mapped_column(
        ForeignKey("cameras.id", ondelete="CASCADE"), index=True
    )
    track_id: Mapped[int] = mapped_column(Integer)
    embedding: Mapped[bytes] = mapped_column(LargeBinary)
    snapshot: Mapped[bytes | None] = mapped_column(LargeBinary, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)


class GlobalEvent(Base):
    """События глобального трекинга: person_seen / camera_transition / person_lost.
    payload — JSON с деталями матча (score, similarity, from/to)."""
    __tablename__ = "global_events"

    id: Mapped[int] = mapped_column(primary_key=True)
    global_id: Mapped[int] = mapped_column(
        ForeignKey("global_persons.id", ondelete="CASCADE"), index=True
    )
    event_type: Mapped[str] = mapped_column(String(30))
    camera_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
    track_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
    payload: Mapped[str | None] = mapped_column(String(2000), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)
