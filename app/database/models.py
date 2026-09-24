"""Модель камеры. Полный RTSP URL в базе НЕ хранится — генерируется из параметров."""
from datetime import datetime, timezone

from sqlalchemy import Boolean, DateTime, Integer, String
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
