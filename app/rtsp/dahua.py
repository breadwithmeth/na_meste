"""Формирование RTSP URL для Dahua NVR.

Стандартный формат Dahua:
    rtsp://USERNAME:PASSWORD@HOST:554/cam/realmonitor?channel=N&subtype=M

    channel — канал камеры на NVR (нумерация с 1)
    subtype=0 — основной поток (main)
    subtype=1 — дополнительный поток (sub)
"""
import re
from urllib.parse import quote


def subtype_code(stream_type: str) -> int:
    """'main' → 0, 'sub' → 1."""
    return 0 if stream_type == "main" else 1


def build_rtsp_url(
    nvr_host: str,
    rtsp_port: int,
    username: str,
    password: str,
    channel: int,
    stream_type: str,
) -> str:
    """Собирает RTSP URL из параметров подключения.

    Логин и пароль URL-кодируются: спецсимволы (@ : # ! % и т.п.)
    не должны ломать URL.
    """
    auth = f"{quote(str(username), safe='')}:{quote(str(password), safe='')}"
    return (
        f"rtsp://{auth}@{nvr_host}:{rtsp_port}"
        f"/cam/realmonitor?channel={channel}&subtype={subtype_code(stream_type)}"
    )


_CREDENTIALS_RE = re.compile(r"(//[^:/@\s]+:)[^@\s]*(@)")


def mask_url(text: str) -> str:
    """Заменяет пароль в RTSP URL на ***. Применяется ко всему, что идёт в логи."""
    return _CREDENTIALS_RE.sub(r"\1***\2", text)
