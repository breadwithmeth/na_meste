"""REST API предметов сцены (еда на столе, посуда, оружие, носимые вещи).

Эндпоинты:

    GET /api/objects          история наблюдений (фильтры: камера, метка,
                              категория)
    GET /api/objects/live     что видно на камерах сейчас (из DetectionState)
"""
import logging

from fastapi import APIRouter, Depends, HTTPException, Request, Response
from sqlalchemy.orm import Session

from app.ai.objects import CATEGORY_LABELS, ru_label
from app.database.database import get_db
from app.database.models import Camera, Employee, ObjectObservation
from app.services.presence_service import iso_local

logger = logging.getLogger("app.api.objects")

router = APIRouter(prefix="/api/objects", tags=["objects"])

LIST_CAP = 500


def _detection_state(request: Request):
    """DetectionState AI-воркера (может отсутствовать, если AI выключен)."""
    return getattr(request.app.state, "detection_state", None)


@router.get("")
def api_objects_history(
    camera_id: int | None = None,
    label: str | None = None,
    category: str | None = None,
    limit: int = 200,
    db: Session = Depends(get_db),
):
    """История наблюдений предметов, свежие сверху (снимки оружия — по id)."""
    limit = max(1, min(limit, LIST_CAP))
    query = (
        db.query(ObjectObservation, Camera.name, Employee.name)
        .outerjoin(Camera, ObjectObservation.camera_id == Camera.id)
        .outerjoin(Employee, ObjectObservation.employee_id == Employee.id)
    )
    if camera_id is not None:
        query = query.filter(ObjectObservation.camera_id == camera_id)
    if label:
        query = query.filter(ObjectObservation.label == label)
    if category:
        query = query.filter(ObjectObservation.category == category)

    rows = (
        query.order_by(ObjectObservation.created_at.desc(), ObjectObservation.id.desc())
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
            "label": obs.label,
            "label_ru": ru_label(obs.label),
            "category": obs.category,
            "category_label": CATEGORY_LABELS.get(obs.category, obs.category),
            "confidence": obs.confidence,
            "has_snapshot": obs.snapshot is not None,
            "created_at": iso_local(obs.created_at),
        }
        for obs, camera_name, employee_name in rows
    ]


@router.get("/{obs_id}/snapshot")
def api_object_snapshot(obs_id: int, db: Session = Depends(get_db)):
    """JPEG-снимок предмета (сохраняется только для оружия)."""
    obs = db.get(ObjectObservation, obs_id)
    if obs is None or obs.snapshot is None:
        raise HTTPException(status_code=404, detail="снимок не найден")
    return Response(content=obs.snapshot, media_type="image/jpeg")


@router.get("/live")
def api_objects_live(request: Request, db: Session = Depends(get_db)):
    """Что видно на камерах прямо сейчас — из последних результатов AI."""
    state = _detection_state(request)
    if state is None:
        return []
    camera_names = {c.id: c.name for c in db.query(Camera).all()}
    result = []
    for camera in state.snapshot():
        for obj in camera.get("objects", []):
            result.append({
                "camera_id": camera["camera_id"],
                "camera_name": camera_names.get(
                    camera["camera_id"], f"камера #{camera['camera_id']}"),
                **obj,
                "category_label": CATEGORY_LABELS.get(obj["category"], obj["category"]),
            })
    return result
