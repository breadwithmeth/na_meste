"""REST API распознавания действий (сидит/работает/отдыхает/кушает…).

Эндпоинты:

    GET /api/actions             история наблюдений (фильтры: камера,
                                 сотрудник, глобальная личность, действие)
    GET /api/actions/live        текущие действия людей на камерах (из
                                 DetectionState AI-воркера)
    GET /api/actions/summary     сводка за период: грубая длительность
                                 действий по сотрудникам/личностям
"""
import logging
from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, Depends, Request
from sqlalchemy import func
from sqlalchemy.orm import Session

from app.ai.actions import ACTION_LABELS
from app.config import settings
from app.database.database import get_db
from app.database.models import ActionObservation, Camera, Employee
from app.services.action_service import ACTION_ORDER
from app.services.presence_service import iso_local, utcnow

logger = logging.getLogger("app.api.actions")

router = APIRouter(prefix="/api/actions", tags=["actions"])

LIST_CAP = 500


def _detection_state(request: Request):
    """DetectionState AI-воркера (может отсутствовать, если AI выключен)."""
    return getattr(request.app.state, "detection_state", None)


def _iso_from_epoch(epoch: float | None) -> str | None:
    """epoch-секунды → локальное ISO (action_since хранится как epoch)."""
    if not epoch:
        return None
    return datetime.fromtimestamp(epoch, tz=timezone.utc).astimezone().isoformat(
        timespec="seconds")


@router.get("")
def api_actions_history(
    camera_id: int | None = None,
    employee_id: int | None = None,
    global_id: int | None = None,
    action: str | None = None,
    limit: int = 200,
    db: Session = Depends(get_db),
):
    """История наблюдений действий, свежие сверху (время — локальное ISO)."""
    limit = max(1, min(limit, LIST_CAP))
    query = (
        db.query(ActionObservation, Camera.name, Employee.name)
        .outerjoin(Camera, ActionObservation.camera_id == Camera.id)
        .outerjoin(Employee, ActionObservation.employee_id == Employee.id)
    )
    if camera_id is not None:
        query = query.filter(ActionObservation.camera_id == camera_id)
    if employee_id is not None:
        query = query.filter(ActionObservation.employee_id == employee_id)
    if global_id is not None:
        query = query.filter(ActionObservation.global_id == global_id)
    if action:
        query = query.filter(ActionObservation.action == action)

    rows = (
        query.order_by(ActionObservation.created_at.desc(), ActionObservation.id.desc())
        .limit(limit)
        .all()
    )
    return [
        {
            "id": obs.id,
            "camera_id": obs.camera_id,
            "camera_name": camera_name or f"камера #{obs.camera_id}",
            "track_id": obs.track_id,
            "employee_id": obs.employee_id,
            "employee_name": employee_name,
            "global_id": obs.global_id,
            "action": obs.action,
            "action_label": ACTION_LABELS.get(obs.action, obs.action),
            "confidence": obs.confidence,
            "created_at": iso_local(obs.created_at),
        }
        for obs, camera_name, employee_name in rows
    ]


@router.get("/live")
def api_actions_live(request: Request, db: Session = Depends(get_db)):
    """Кто что делает прямо сейчас — из последних результатов AI-воркера."""
    state = _detection_state(request)
    if state is None:
        return []
    camera_names = {c.id: c.name for c in db.query(Camera).all()}
    result = []
    for camera in state.snapshot():
        for track in camera.get("tracks", []):
            if track.get("action") is None:
                continue
            result.append({
                "camera_id": camera["camera_id"],
                "camera_name": camera_names.get(
                    camera["camera_id"], f"камера #{camera['camera_id']}"),
                "track_id": track["track_id"],
                "employee_id": track.get("employee_id"),
                "employee_name": track.get("employee_name"),
                "global_id": track.get("global_id"),
                "action": track["action"],
                "action_label": ACTION_LABELS.get(track["action"], track["action"]),
                "action_confidence": track.get("action_confidence"),
                "action_since": _iso_from_epoch(track.get("action_since")),
            })
    return result


@router.get("/summary")
def api_actions_summary(
    hours: float = 8.0,
    employee_id: int | None = None,
    camera_id: int | None = None,
    db: Session = Depends(get_db),
):
    """Сводка за период: грубая длительность действий (число наблюдений ×
    ACTION_LOG_INTERVAL) по сотрудникам и неопознанным личностям."""
    hours = max(0.1, min(hours, 720.0))
    cutoff = utcnow() - timedelta(hours=hours)
    query = db.query(
        ActionObservation.employee_id,
        ActionObservation.global_id,
        ActionObservation.action,
        func.count(ActionObservation.id),
    ).filter(ActionObservation.created_at >= cutoff)
    if employee_id is not None:
        query = query.filter(ActionObservation.employee_id == employee_id)
    if camera_id is not None:
        query = query.filter(ActionObservation.camera_id == camera_id)
    rows = query.group_by(
        ActionObservation.employee_id,
        ActionObservation.global_id,
        ActionObservation.action,
    ).all()

    employee_names = {e.id: e.name for e in db.query(Employee).all()}
    step = max(1.0, settings.action_log_interval)

    persons: dict[tuple, dict] = {}
    for emp_id, gid, action, count in rows:
        key = ("employee", emp_id) if emp_id is not None else ("global", gid)
        person = persons.setdefault(key, {
            "employee_id": emp_id,
            "employee_name": employee_names.get(emp_id),
            "global_id": gid,
            "actions": {},
            "total_seconds": 0.0,
        })
        seconds = round(count * step)
        person["actions"][action] = seconds
        person["total_seconds"] += seconds

    result = sorted(persons.values(), key=lambda p: -p["total_seconds"])
    for person in result:
        # стабильный порядок действий: working → sitting → … → lying
        person["actions"] = {
            a: person["actions"][a]
            for a in ACTION_ORDER + sorted(set(person["actions"]) - set(ACTION_ORDER))
            if a in person["actions"]
        }
        person["actions_label"] = {
            a: ACTION_LABELS.get(a, a) for a in person["actions"]
        }
    return result
