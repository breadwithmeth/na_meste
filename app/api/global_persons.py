"""REST API глобальных персон (межкамерный трекинг)."""
import json

from fastapi import APIRouter, Depends, HTTPException, Response
from sqlalchemy import func
from sqlalchemy.orm import Session

from app.database.database import get_db
from app.database.models import (
    Camera, Employee, GlobalEvent, GlobalObservation, GlobalPerson,
)
from app.services.presence_service import iso_local

router = APIRouter(prefix="/api/global-persons", tags=["global-persons"])


def _person_or_404(db: Session, global_id: int) -> GlobalPerson:
    person = db.get(GlobalPerson, global_id)
    if person is None:
        raise HTTPException(status_code=404, detail="Global person not found")
    return person


def _names(db: Session):
    cameras = {c.id: c.name for c in db.query(Camera).all()}
    employees = {e.id: e.name for e in db.query(Employee).all()}
    return cameras, employees


def _person_row(db: Session, person: GlobalPerson) -> dict:
    cameras, employees = _names(db)
    observations = db.query(func.count(GlobalObservation.id)).filter(
        GlobalObservation.global_id == person.id
    ).scalar() or 0
    return {
        "global_id": person.id,
        "status": person.status,
        "employee_id": person.employee_id,
        "employee_name": employees.get(person.employee_id),
        "last_camera_id": person.last_camera_id,
        "last_camera_name": cameras.get(person.last_camera_id),
        "last_track_id": person.last_track_id,
        "observations_count": observations,
        "created_at": iso_local(person.created_at),
        "last_seen_at": iso_local(person.last_seen_at),
    }


@router.get("")
def api_list_global_persons(limit: int = 100, db: Session = Depends(get_db)):
    """Глобальные личности, свежие сверху."""
    limit = max(1, min(limit, 500))
    persons = db.query(GlobalPerson).order_by(
        GlobalPerson.last_seen_at.desc()
    ).limit(limit).all()
    return [_person_row(db, p) for p in persons]


@router.get("/{global_id}")
def api_get_global_person(global_id: int, db: Session = Depends(get_db)):
    return _person_row(db, _person_or_404(db, global_id))


@router.get("/{global_id}/timeline")
def api_global_timeline(global_id: int, limit: int = 200, db: Session = Depends(get_db)):
    """Хронология: наблюдения + события (переходы и т.п.), свежие сверху."""
    _person_or_404(db, global_id)
    cameras, employees = _names(db)
    items = []
    observations = db.query(GlobalObservation).filter(
        GlobalObservation.global_id == global_id
    ).order_by(GlobalObservation.created_at.desc()).limit(limit).all()
    for o in observations:
        items.append({
            "kind": "observation",
            "id": o.id,
            "camera_id": o.camera_id,
            "camera_name": cameras.get(o.camera_id),
            "track_id": o.track_id,
            "has_snapshot": bool(o.snapshot),
            "created_at": iso_local(o.created_at),
        })
    events = db.query(GlobalEvent).filter(
        GlobalEvent.global_id == global_id
    ).order_by(GlobalEvent.created_at.desc()).limit(limit).all()
    for e in events:
        items.append({
            "kind": "event",
            "id": e.id,
            "event_type": e.event_type,
            "camera_id": e.camera_id,
            "camera_name": cameras.get(e.camera_id),
            "track_id": e.track_id,
            "payload": json.loads(e.payload) if e.payload else {},
            "created_at": iso_local(e.created_at),
        })
    items.sort(key=lambda x: x["created_at"], reverse=True)
    return items


@router.get("/{global_id}/observations")
def api_global_observations(global_id: int, limit: int = 200,
                             db: Session = Depends(get_db)):
    _person_or_404(db, global_id)
    cameras, _employees = _names(db)
    observations = db.query(GlobalObservation).filter(
        GlobalObservation.global_id == global_id
    ).order_by(GlobalObservation.created_at.desc()).limit(limit).all()
    return [
        {
            "id": o.id,
            "camera_id": o.camera_id,
            "camera_name": cameras.get(o.camera_id),
            "track_id": o.track_id,
            "has_snapshot": bool(o.snapshot),
            "created_at": iso_local(o.created_at),
        }
        for o in observations
    ]


@router.get("/{global_id}/observations/{observation_id}/photo")
def api_observation_photo(global_id: int, observation_id: int,
                          db: Session = Depends(get_db)):
    observation = db.get(GlobalObservation, observation_id)
    if observation is None or observation.global_id != global_id:
        raise HTTPException(status_code=404, detail="Observation not found")
    if not observation.snapshot:
        raise HTTPException(status_code=404, detail="Snapshot not found")
    return Response(content=observation.snapshot, media_type="image/jpeg")


@router.get("/{global_id}/trajectory")
def api_global_trajectory(global_id: int, db: Session = Depends(get_db)):
    """Траектория: последовательность камер по времени + переходы между ними."""
    _person_or_404(db, global_id)
    cameras, _employees = _names(db)
    observations = db.query(GlobalObservation).filter(
        GlobalObservation.global_id == global_id
    ).order_by(GlobalObservation.created_at.asc()).all()

    segments: list[dict] = []
    for o in observations:
        name = cameras.get(o.camera_id, f"#{o.camera_id}")
        if segments and segments[-1]["camera_id"] == o.camera_id:
            segments[-1]["to"] = iso_local(o.created_at)
            segments[-1]["observations"] += 1
        else:
            segments.append({
                "camera_id": o.camera_id,
                "camera_name": name,
                "from": iso_local(o.created_at),
                "to": iso_local(o.created_at),
                "observations": 1,
            })
    return {
        "global_id": global_id,
        "segments": segments,
        "chain": [s["camera_name"] for s in segments],
    }
