"""События предметов сцены (еда на столе, посуда, оружие, носимые вещи):
БД → Telegram (только оружие, с фото).

Вызывается слоем предметов один раз на подтверждённый объект; здесь —
кулдаун на камеру+метку (повторные появления не спамят), JPEG-кроп
(для оружия — в БД и Telegram) и очистка старых записей.
"""
import logging
import time
from datetime import datetime, timedelta

import cv2

from app.ai.objects import SceneObject
from app.database.models import Camera, ObjectObservation
from app.services.presence_service import utcnow

logger = logging.getLogger("app.services.objects")

SNAPSHOT_MAX_HEIGHT = 720   # px, для снимка оружия в БД и Telegram


class ObjectEventManager:
    def __init__(
        self,
        session_factory,
        notifier=None,
        keep_days: int = 30,
        weapon_alert_enabled: bool = True,
        weapon_alert_cooldown: float = 60.0,
        event_cooldown: float = 30.0,
    ):
        self._session_factory = session_factory
        self.notifier = notifier
        self.keep_days = keep_days
        self.weapon_alert_enabled = weapon_alert_enabled
        self.weapon_alert_cooldown = weapon_alert_cooldown
        self.event_cooldown = event_cooldown
        self._last_event: dict[tuple[int, str], float] = {}

    def record(self, camera_id: int, obj: SceneObject, frame,
               person: dict | None) -> None:
        """Подтверждённый объект → БД; оружие → фото в Telegram.
        Ошибки не должны ломать AI-воркер."""
        now = time.monotonic()
        key = (camera_id, obj.label)
        if now - self._last_event.get(key, 0.0) < self.event_cooldown:
            return
        self._last_event[key] = now

        try:
            snapshot = None
            if obj.category == "weapon":
                snapshot = self._crop_object(frame, obj.bbox)
            person = person or {}
            with self._session_factory() as db:
                db.add(ObjectObservation(
                    camera_id=camera_id,
                    track_id=person.get("track_id"),
                    employee_id=person.get("employee_id"),
                    global_id=person.get("global_id"),
                    label=obj.label,
                    category=obj.category,
                    confidence=obj.confidence,
                    snapshot=snapshot,
                ))
                db.commit()
            logger.info(
                "Camera %d: предмет %s (%s) закреплён%s",
                camera_id, obj.label, obj.category,
                " — ОРУЖИЕ" if obj.category == "weapon" else "",
            )
            if (obj.category == "weapon" and snapshot is not None
                    and self.weapon_alert_enabled and self.notifier is not None):
                with self._session_factory() as db:
                    camera = db.get(Camera, camera_id)
                    camera_name = camera.name if camera else f"камера #{camera_id}"
                self.notifier.send_photo(
                    snapshot, self._caption(camera_name, obj, person))
        except Exception:
            logger.exception(
                "Objects: не удалось записать событие (cam=%s, %s)",
                camera_id, obj.label,
            )

    def cleanup(self) -> None:
        """Удалить наблюдения старше object_keep_days."""
        cutoff = utcnow() - timedelta(days=self.keep_days)
        try:
            with self._session_factory() as db:
                db.query(ObjectObservation).filter(
                    ObjectObservation.created_at < cutoff
                ).delete()
                db.commit()
        except Exception:
            logger.exception("Objects: ошибка очистки старых записей")

    # ------------------------------------------------------------ внутреннее

    @staticmethod
    def _crop_object(frame, bbox_norm) -> bytes:
        """Кроп объекта (норм. bbox, запас 20%) → JPEG."""
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = (float(v) for v in bbox_norm)
        px1, py1, px2, py2 = int(x1 * w), int(y1 * h), int(x2 * w), int(y2 * h)
        mx, my = int((px2 - px1) * 0.2), int((py2 - py1) * 0.2)
        px1, py1 = max(0, px1 - mx), max(0, py1 - my)
        px2, py2 = min(w, px2 + mx), min(h, py2 + my)
        crop = frame[max(0, py1):py2, max(0, px1):px2]
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
    def _caption(camera_name: str, obj: SceneObject, person: dict) -> str:
        local = datetime.now().strftime("%d.%m.%Y %H:%M:%S")
        who = person.get("employee_name") or "посторонний"
        gid = f" (G#{person['global_id']})" if person.get("global_id") else ""
        return (
            f"🚨 Обнаружено оружие: {obj.label_ru}\n"
            f"Кем: {who}{gid}\n"
            f"Камера: {camera_name}\nВремя: {local}"
        )
