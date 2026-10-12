"""REST API внешности людей (цвет одежды, пол, возраст).

Эндпоинты:

    GET /api/appearance         история наблюдений (фильтры: камера,
                                сотрудник, глобальная личность, цвет, пол)
    GET /api/appearance/latest  последний известный облик по каждому человеку
"""
import logging

from fastapi import APIRouter, Depends
from sqlalchemy import func
from sqlalchemy.orm import Session

from app.database.database import get_db
from app.database.models import AppearanceObservation, Camera, Employee
from app.services.presence_service import iso_local

logger = logging.getLogger("app.api.appearance")

router = APIRouter(prefix="/api/appearance", tags=["appearance"])

LIST_CAP = 500


def _serialize(obs: AppearanceObservation, camera_name, employee_name) -> dict:
    return {
        "id": obs.id,
        "camera_id": obs.camera_id,
        "camera_name": camera_name or f"камера #{obs.camera_id}",
        "track_id": obs.track_id,
        "employee_id": obs.employee_id,
        "employee_name": employee_name,
        "global_id": obs.global_id,
        "upper_color": obs.upper_color,
        "lower_color": obs.lower_color,
        "gender": obs.gender,
        "age": obs.age,
        "created_at": iso_local(obs.created_at),
    }


@router.get("")
def api_appearance_history(
    camera_id: int | None = None,
    employee_id: int | None = None,
    global_id: int | None = None,
    color: str | None = None,
    gender: str | None = None,
    limit: int = 200,
    db: Session = Depends(get_db),
):
    """История наблюдений внешности, свежие сверху. color ищет и в верхней,
    и в нижней одежде («синий» найдёт синюю куртку и синие брюки)."""
    limit = max(1, min(limit, LIST_CAP))
    query = (
        db.query(AppearanceObservation, Camera.name, Employee.name)
        .outerjoin(Camera, AppearanceObservation.camera_id == Camera.id)
        .outerjoin(Employee, AppearanceObservation.employee_id == Employee.id)
    )
    if camera_id is not None:
        query = query.filter(AppearanceObservation.camera_id == camera_id)
    if employee_id is not None:
        query = query.filter(AppearanceObservation.employee_id == employee_id)
    if global_id is not None:
        query = query.filter(AppearanceObservation.global_id == global_id)
    if gender:
        query = query.filter(AppearanceObservation.gender == gender)
    if color:
        query = query.filter(
            (AppearanceObservation.upper_color == color)
            | (AppearanceObservation.lower_color == color)
        )

    rows = (
        query.order_by(
            AppearanceObservation.created_at.desc(), AppearanceObservation.id.desc()
        )
        .limit(limit)
        .all()
    )
    return [_serialize(obs, camera_name, employee_name)
            for obs, camera_name, employee_name in rows]


@router.get("/latest")
def api_appearance_latest(
    employee_id: int | None = None,
    db: Session = Depends(get_db),
):
    """Последний известный облик по каждому человеку (сотрудники и
    глобальные личности): когда и в чём видели последний раз."""
    query = db.query(
        AppearanceObservation.employee_id,
        AppearanceObservation.global_id,
        func.max(AppearanceObservation.created_at),
    ).group_by(
        AppearanceObservation.employee_id,
        AppearanceObservation.global_id,
    )
    if employee_id is not None:
        query = query.filter(AppearanceObservation.employee_id == employee_id)

    latest = {}
    for emp_id, gid, last_seen in query.all():
        key = ("employee", emp_id) if emp_id is not None else ("global", gid)
        latest[key] = (emp_id, gid, last_seen)

    employee_names = {e.id: e.name for e in db.query(Employee).all()}
    camera_names = {c.id: c.name for c in db.query(Camera).all()}

    result = []
    for emp_id, gid, last_seen in latest.values():
        obs = (
            db.query(AppearanceObservation, Camera.name)
            .outerjoin(Camera, AppearanceObservation.camera_id == Camera.id)
            .filter(
                AppearanceObservation.created_at == last_seen,
                AppearanceObservation.employee_id == emp_id,
                AppearanceObservation.global_id == gid,
            )
            .order_by(AppearanceObservation.id.desc())
            .first()
        )
        if obs is None:
            continue
        item = _serialize(obs[0], obs[1], employee_names.get(emp_id))
        if emp_id is None:
            item["employee_name"] = None
        result.append(item)

    result.sort(key=lambda item: item["created_at"] or "", reverse=True)
    return result
