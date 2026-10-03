"""REST API сотрудников и их лиц."""
import logging

from fastapi import APIRouter, Depends, File, HTTPException, Request, UploadFile
from fastapi.responses import Response
from sqlalchemy import func
from sqlalchemy.orm import Session

from app.database.database import get_db
from app.database.models import EmployeeFace
from app.services import employee_service as service

logger = logging.getLogger("app.api.employees")

router = APIRouter(prefix="/api/employees", tags=["employees"])

MAX_PHOTO_BYTES = 15 * 1024 * 1024  # 15 МБ на файл


def _face_engine_or_503(request: Request):
    engine = getattr(request.app.state, "face_engine", None)
    if engine is None:
        raise HTTPException(
            status_code=503,
            detail="AI-модели недоступны — загрузка фотографий невозможна (см. README)",
        )
    return engine


def _store_or_503(request: Request):
    store = getattr(request.app.state, "embedding_store", None)
    if store is None:
        raise HTTPException(status_code=503, detail="AI-хранилище эмбеддингов недоступно")
    return store


def _employee_or_404(db: Session, employee_id: int):
    employee = service.get_employee(db, employee_id)
    if employee is None:
        raise HTTPException(status_code=404, detail="Employee not found")
    return employee


def _faces_count(db: Session, employee_id: int) -> int:
    return db.query(func.count(EmployeeFace.id)).filter(
        EmployeeFace.employee_id == employee_id
    ).scalar() or 0


def _employee_out(db: Session, employee) -> dict:
    data = service.EmployeeOut.model_validate(employee).model_dump(mode="json")
    data["faces_count"] = _faces_count(db, employee.id)
    return data


@router.get("")
def api_list_employees(db: Session = Depends(get_db)):
    return [_employee_out(db, e) for e, _count in service.list_employees(db)]


@router.post("", status_code=201)
def api_create_employee(
    payload: service.EmployeeCreate,
    request: Request,
    db: Session = Depends(get_db),
):
    employee = service.create_employee(db, payload)
    store = getattr(request.app.state, "embedding_store", None)
    if store:
        store.refresh(db)
    return _employee_out(db, employee)


@router.get("/{employee_id}")
def api_get_employee(employee_id: int, faces: bool = False, db: Session = Depends(get_db)):
    """Карточка сотрудника; faces=true — ещё и список его лиц (без эмбеддингов)."""
    employee = _employee_or_404(db, employee_id)
    data = _employee_out(db, employee)
    if faces:
        data["faces"] = [
            {"id": f.id, "created_at": f.created_at.isoformat(timespec="seconds")}
            for f in db.query(EmployeeFace)
            .filter(EmployeeFace.employee_id == employee_id)
            .order_by(EmployeeFace.id.desc()).all()
        ]
    return data


@router.put("/{employee_id}")
def api_update_employee(
    employee_id: int,
    payload: service.EmployeeUpdate,
    request: Request,
    db: Session = Depends(get_db),
):
    employee = _employee_or_404(db, employee_id)
    employee = service.update_employee(db, employee, payload)
    store = getattr(request.app.state, "embedding_store", None)
    if store:
        store.refresh(db)  # active-флаг влияет на набор эмбеддингов
    return _employee_out(db, employee)


@router.delete("/{employee_id}", status_code=204)
def api_delete_employee(
    employee_id: int,
    request: Request,
    db: Session = Depends(get_db),
):
    employee = _employee_or_404(db, employee_id)
    service.delete_employee(db, employee, None)
    store = getattr(request.app.state, "embedding_store", None)
    if store:
        store.refresh(db)
    return None


@router.post("/{employee_id}/faces")
async def api_upload_faces(
    employee_id: int,
    request: Request,
    files: list[UploadFile] = File(...),
    db: Session = Depends(get_db),
):
    """Загрузка фото лица сотрудника: детекция → эмбеддинг → сохранение.

    Результат возвращается по каждому файлу отдельно (успех/ошибка/предупреждения).
    """
    employee = _employee_or_404(db, employee_id)
    engine = _face_engine_or_503(request)
    store = _store_or_503(request)

    results = []
    for upload in files:
        filename = upload.filename or "photo"
        content = await upload.read()
        if not content:
            results.append(service.FaceEnrollResult(
                filename=filename, success=False, error="Пустой файл"))
            continue
        if len(content) > MAX_PHOTO_BYTES:
            results.append(service.FaceEnrollResult(
                filename=filename, success=False,
                error="Файл слишком большой (максимум 15 МБ)"))
            continue
        results.append(service.enroll_face(
            db, employee, content, engine, store, filename=filename
        ))
    return {"results": [r.model_dump() for r in results]}


@router.get("/{employee_id}/faces/{face_id}/thumbnail")
def api_face_thumbnail(employee_id: int, face_id: int, db: Session = Depends(get_db)):
    face = service.get_face(db, employee_id, face_id)
    if face is None or not face.thumbnail:
        raise HTTPException(status_code=404, detail="Thumbnail not found")
    return Response(content=face.thumbnail, media_type="image/jpeg")


@router.delete("/{employee_id}/faces/{face_id}", status_code=204)
def api_delete_face(
    employee_id: int,
    face_id: int,
    request: Request,
    db: Session = Depends(get_db),
):
    _employee_or_404(db, employee_id)
    store = _store_or_503(request)
    face = service.get_face(db, employee_id, face_id)
    if face is None:
        raise HTTPException(status_code=404, detail="Face not found")
    service.delete_face(db, face, store)
    return None
