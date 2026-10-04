"""REST API фиксации посторонних (неопознанных людей)."""
from fastapi import APIRouter, Depends, HTTPException, Response
from sqlalchemy.orm import Session

from app.database.database import get_db
from app.database.models import Camera, UnknownEvent
from app.services.presence_service import iso_local

router = APIRouter(prefix="/api/unknown", tags=["unknown"])


def _event_row(event: UnknownEvent, camera_name: str | None) -> dict:
    return {
        "id": event.id,
        "camera_id": event.camera_id,
        "camera_name": camera_name,
        "track_id": event.track_id,
        "global_id": event.global_id,
        "created_at": iso_local(event.created_at),
    }


@router.get("")
def api_list_unknown(limit: int = 100, global_id: int | None = None,
                     db: Session = Depends(get_db)):
    """Последние фиксации посторонних (свежие сверху).

    global_id — фильтр по конкретной глобальной личности (трекинг постороннего).
    """
    limit = max(1, min(limit, 500))
    query = (
        db.query(UnknownEvent, Camera.name)
        .join(Camera, UnknownEvent.camera_id == Camera.id, isouter=True)
        .order_by(UnknownEvent.created_at.desc())
        .limit(limit)
    )
    if global_id is not None:
        query = query.filter(UnknownEvent.global_id == global_id)
    return [_event_row(event, cam_name) for event, cam_name in query.all()]


@router.get("/{event_id}/photo")
def api_unknown_photo(event_id: int, db: Session = Depends(get_db)):
    event = db.get(UnknownEvent, event_id)
    if event is None or not event.snapshot:
        raise HTTPException(status_code=404, detail="Event not found")
    return Response(content=event.snapshot, media_type="image/jpeg")


@router.delete("/{event_id}", status_code=204)
def api_delete_unknown(event_id: int, db: Session = Depends(get_db)):
    event = db.get(UnknownEvent, event_id)
    if event is None:
        raise HTTPException(status_code=404, detail="Event not found")
    db.delete(event)
    db.commit()
    return None
