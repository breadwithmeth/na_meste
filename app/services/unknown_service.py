"""Фиксация посторонних (неопознанных людей): снимок → БД → Telegram."""
import logging
from datetime import datetime, timedelta

import cv2

from app.database.models import Camera, UnknownEvent
from app.services.presence_service import utcnow

logger = logging.getLogger("app.services.unknown")

SNAPSHOT_MAX_HEIGHT = 720   # px, для снимка в БД и Telegram


class UnknownEventsManager:
    def __init__(self, session_factory, notifier=None, keep_days: int = 30):
        self._session_factory = session_factory
        self.notifier = notifier
        self.keep_days = keep_days

    def record(self, camera_id: int, track_id: int, frame, bbox) -> None:
        """Кроп человека из кадра → JPEG → событие в БД → фото в Telegram.

        Ошибки не должны ломать AI-воркер — ловим и логируем.
        """
        try:
            snapshot = self._crop_person(frame, bbox)
            with self._session_factory() as db:
                camera = db.get(Camera, camera_id)
                camera_name = camera.name if camera else f"камера #{camera_id}"
                event = UnknownEvent(
                    camera_id=camera_id, track_id=track_id, snapshot=snapshot
                )
                db.add(event)
                db.commit()
            logger.info(
                "Camera %d: неизвестный человек зафиксирован track=%d", camera_id, track_id
            )
            if self.notifier is not None:
                self.notifier.send_photo(snapshot, self._caption(camera_name))
        except Exception:
            logger.exception("Unknown: не удалось зафиксировать событие (cam=%s)", camera_id)

    def cleanup(self) -> None:
        """Удалить события старше unknown_keep_days."""
        cutoff = utcnow() - timedelta(days=self.keep_days)
        try:
            with self._session_factory() as db:
                db.query(UnknownEvent).filter(
                    UnknownEvent.created_at < cutoff
                ).delete()
                db.commit()
        except Exception:
            logger.exception("Unknown: ошибка очистки старых событий")

    # ------------------------------------------------------------ внутреннее

    @staticmethod
    def _crop_person(frame, bbox) -> bytes:
        """Кроп человека с запасом 10%, ограничение по высоте, JPEG."""
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = (int(v) for v in bbox)
        mx, my = int((x2 - x1) * 0.1), int((y2 - y1) * 0.1)
        x1, y1 = max(0, x1 - mx), max(0, y1 - my)
        x2, y2 = min(w, x2 + mx), min(h, y2 + my)
        crop = frame[max(0, y1):y2, max(0, x1):x2]
        if crop.size == 0:
            raise ValueError("пустой кроп")
        if crop.shape[0] > SNAPSHOT_MAX_HEIGHT:
            scale = SNAPSHOT_MAX_HEIGHT / crop.shape[0]
            crop = cv2.resize(crop, (int(crop.shape[1] * scale), SNAPSHOT_MAX_HEIGHT))
        ok, buf = cv2.imencode(".jpg", crop, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        if not ok:
            raise ValueError("не удалось закодировать JPEG")
        return buf.tobytes()

    @staticmethod
    def _caption(camera_name: str) -> str:
        local = datetime.now().strftime("%d.%m.%Y %H:%M:%S")
        return f"⚠️ Посторонний\nКамера: {camera_name}\nВремя: {local}"
