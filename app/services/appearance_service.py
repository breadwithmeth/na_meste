"""История внешности людей (цвет одежды, пол, возраст) → БД.

Пишется слоем атрибутов (AttributeExtractor) при смене облика и далее
с троттлингом ATTRIBUTES_LOG_INTERVAL, пока облик не меняется.
"""
import logging
from datetime import timedelta

from app.database.models import AppearanceObservation
from app.services.presence_service import utcnow

logger = logging.getLogger("app.services.appearance")


class AppearanceLogManager:
    def __init__(self, session_factory, keep_days: int = 30):
        self._session_factory = session_factory
        self.keep_days = keep_days

    def record(self, camera_id: int, track_id: int, employee_id,
               global_id, upper_color, lower_color, gender, age) -> None:
        """Наблюдение внешности → БД. Ошибки не ломают AI-воркер."""
        try:
            with self._session_factory() as db:
                db.add(AppearanceObservation(
                    camera_id=camera_id,
                    track_id=track_id,
                    employee_id=employee_id,
                    global_id=global_id,
                    upper_color=upper_color,
                    lower_color=lower_color,
                    gender=gender,
                    age=age,
                ))
                db.commit()
        except Exception:
            logger.exception("Appearance: не удалось записать наблюдение")

    def cleanup(self) -> None:
        """Удалить наблюдения старше appearance_keep_days."""
        cutoff = utcnow() - timedelta(days=self.keep_days)
        try:
            with self._session_factory() as db:
                db.query(AppearanceObservation).filter(
                    AppearanceObservation.created_at < cutoff
                ).delete()
                db.commit()
        except Exception:
            logger.exception("Appearance: ошибка очистки старых записей")
