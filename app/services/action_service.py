"""Запись наблюдений действий (action_observations) и очистка старых.

Действие классифицируется на треке (см. app/ai/actions.py); менеджер —
тонкий писатель в БД: сам решает, когда писать (смена действия или
троттлинг ACTION_LOG_INTERVAL при неизменном), ошибки не ломают воркер.
"""
import logging
from datetime import timedelta

from app.database.models import ActionObservation
from app.services.presence_service import utcnow

logger = logging.getLogger("app.services.actions")

# действия, показываемые в UI и API (порядок = порядок таблицы сводки)
ACTION_ORDER = [
    "working", "sitting", "standing", "walking",
    "resting", "eating", "phone", "lying",
]


class ActionLogManager:
    def __init__(self, session_factory, keep_days: int = 30):
        self._session_factory = session_factory
        self.keep_days = keep_days

    def record(
        self, camera_id: int, track_id: int, employee_id: int | None,
        global_id: int | None, action: str, confidence: float | None,
    ) -> None:
        """Одно наблюдение действия. Вызывается из AI-воркера."""
        try:
            with self._session_factory() as db:
                db.add(ActionObservation(
                    camera_id=camera_id,
                    track_id=track_id,
                    employee_id=employee_id,
                    global_id=global_id,
                    action=action,
                    confidence=confidence,
                ))
                db.commit()
        except Exception:
            logger.exception(
                "Не удалось записать действие (cam=%s track=%s)", camera_id, track_id)

    def cleanup(self) -> None:
        """Удалить наблюдения старнее action_keep_days."""
        cutoff = utcnow() - timedelta(days=self.keep_days)
        try:
            with self._session_factory() as db:
                db.query(ActionObservation).filter(
                    ActionObservation.created_at < cutoff
                ).delete()
                db.commit()
        except Exception:
            logger.exception("Действия: ошибка очистки старых наблюдений")
