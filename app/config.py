"""Конфигурация приложения (читается из .env в корне проекта)."""
import os
import shutil
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

# Корень проекта — папка, в которой лежит main.py
BASE_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(BASE_DIR / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_host: str = "0.0.0.0"
    app_port: int = 8000

    database_url: str = "sqlite:///./data/app.db"

    rtsp_connect_timeout: int = 10          # сек, таймаут подключения к RTSP
    rtsp_reconnect_max_delay: int = 30      # сек, потолок exponential backoff
    default_rtsp_port: int = 554

    ffmpeg_path: str = ""                   # пусто — автоопределение
    preview_fps: int = 10                   # ограничение FPS MJPEG-превью
    frame_buffer_size: int = 2              # буфер кадров (drop old, keep latest)
    frame_stall_timeout: int = 10           # сек без кадров → принудительный реконнект


settings = Settings()


def _exe(name: str) -> str:
    return f"{name}.exe" if os.name == "nt" else name


def resolve_ffmpeg() -> tuple[str | None, str | None]:
    """Ищет ffmpeg/ffprobe: FFMPEG_PATH из .env → bin/ проекта → системный PATH.

    Возвращает (путь_к_ffmpeg, путь_к_ffprobe) или (None, None).
    """
    candidates: list[tuple[Path, Path]] = []

    if settings.ffmpeg_path:
        ffmpeg = Path(settings.ffmpeg_path)
        candidates.append((ffmpeg, ffmpeg.parent / _exe("ffprobe")))

    local_bin = BASE_DIR / "bin"
    candidates.append((local_bin / _exe("ffmpeg"), local_bin / _exe("ffprobe")))

    for ffmpeg, ffprobe in candidates:
        if ffmpeg.is_file() and ffprobe.is_file():
            return str(ffmpeg), str(ffprobe)

    ffmpeg = shutil.which("ffmpeg")
    ffprobe = shutil.which("ffprobe")
    if ffmpeg and ffprobe:
        return ffmpeg, ffprobe
    return None, None
