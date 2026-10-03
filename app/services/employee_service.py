"""CRUD сотрудников и зачисление лиц (фото → эмбеддинг → SQLite)."""
import logging
from datetime import datetime
from typing import Optional

import cv2
import numpy as np
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import func
from sqlalchemy.orm import Session

from app.ai.face_detector import FaceEngine
from app.ai.face_recognition import EmbeddingStore
from app.config import settings
from app.database.models import Employee, EmployeeFace, PresenceSession

logger = logging.getLogger("app.services.employees")


# ------------------------------------------------------------------- схемы

class EmployeeCreate(BaseModel):
    name: str = Field(min_length=1, max_length=150)
    external_id: Optional[str] = Field(default=None, max_length=64)
    active: bool = True


class EmployeeUpdate(BaseModel):
    name: Optional[str] = Field(default=None, min_length=1, max_length=150)
    external_id: Optional[str] = Field(default=None, max_length=64)
    active: Optional[bool] = None


class EmployeeOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    name: str
    external_id: Optional[str]
    active: bool
    faces_count: int = 0
    created_at: datetime
    updated_at: datetime


class FaceEnrollResult(BaseModel):
    filename: str
    success: bool
    face_id: Optional[int] = None
    error: Optional[str] = None
    warnings: list[str] = Field(default_factory=list)


# --------------------------------------------------------------------- CRUD

def get_employee(db: Session, employee_id: int) -> Optional[Employee]:
    return db.get(Employee, employee_id)


def list_employees(db: Session) -> list[tuple[Employee, int]]:
    """Сотрудники + количество лиц каждого."""
    employees = db.query(Employee).order_by(Employee.name).all()
    counts = dict(
        db.query(EmployeeFace.employee_id, func.count(EmployeeFace.id))
        .group_by(EmployeeFace.employee_id).all()
    )
    return [(e, counts.get(e.id, 0)) for e in employees]


def create_employee(db: Session, data: EmployeeCreate) -> Employee:
    employee = Employee(**data.model_dump())
    db.add(employee)
    db.commit()
    db.refresh(employee)
    logger.info("Создан сотрудник #%d «%s»", employee.id, employee.name)
    return employee


def update_employee(db: Session, employee: Employee, data: EmployeeUpdate) -> Employee:
    for field, value in data.model_dump(exclude_unset=True).items():
        setattr(employee, field, value)
    db.commit()
    db.refresh(employee)
    return employee


def delete_employee(db: Session, employee: Employee, store: Optional[EmbeddingStore]) -> None:
    """Удаляет сотрудника вместе с лицами и историей присутствия.

    Эмбеддинг-стор обновляет вызывающий код (у него своя сессия БД).
    """
    db.query(PresenceSession).filter(PresenceSession.employee_id == employee.id).delete()
    db.query(EmployeeFace).filter(EmployeeFace.employee_id == employee.id).delete()
    db.delete(employee)
    db.commit()
    logger.info("Удалён сотрудник #%d «%s»", employee.id, employee.name)


# --------------------------------------------------------- зачисление лица

def enroll_face(
    db: Session,
    employee: Employee,
    image_bytes: bytes,
    face_engine: FaceEngine,
    store: EmbeddingStore,
    filename: str = "photo",
) -> FaceEnrollResult:
    """Фото → (детекция лица → проверки) → эмбеддинг → employee_faces.

    Ошибки: нет лица / несколько лиц / лицо слишком маленькое.
    Предупреждения (не мешают сохранению): низкое качество, размытость.
    """
    img = cv2.imdecode(np.frombuffer(image_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return FaceEnrollResult(filename=filename, success=False,
                                error="Не удалось прочитать файл изображения")

    faces = face_engine.detect(img)
    if not faces:
        return FaceEnrollResult(filename=filename, success=False,
                                error="Лицо на фотографии не найдено")
    if len(faces) > 1:
        return FaceEnrollResult(
            filename=filename, success=False,
            error=f"На фотографии {len(faces)} лиц — загрузите фото с одним лицом",
        )

    face = faces[0]
    fw = float(face.bbox[2] - face.bbox[0])
    fh = float(face.bbox[3] - face.bbox[1])
    if min(fw, fh) < settings.face_min_size:
        return FaceEnrollResult(
            filename=filename, success=False,
            error=(f"Лицо слишком маленькое ({fw:.0f}×{fh:.0f}px, "
                   f"минимум {settings.face_min_size}px)"),
        )

    warnings: list[str] = []
    if face.det_score < 0.65:
        warnings.append(f"Низкая уверенность детекции ({face.det_score:.2f})")
    if face.aligned is not None:
        gray = cv2.cvtColor(face.aligned, cv2.COLOR_BGR2GRAY)
        blur = cv2.Laplacian(gray, cv2.CV_64F).var()
        if blur < 20:
            warnings.append("Изображение размыто — качество распознавания может снизиться")

    thumbnail = None
    if face.aligned is not None:
        ok, buf = cv2.imencode(".jpg", face.aligned,
                               [int(cv2.IMWRITE_JPEG_QUALITY), 85])
        if ok:
            thumbnail = buf.tobytes()

    employee_face = EmployeeFace(
        employee_id=employee.id,
        embedding=np.asarray(face.embedding, dtype=np.float32).tobytes(),
        thumbnail=thumbnail,
    )
    db.add(employee_face)
    db.commit()
    db.refresh(employee_face)
    store.refresh(db)
    logger.info("Сотруднику #%d добавлено лицо (face_id=%d, предупреждений: %d)",
                employee.id, employee_face.id, len(warnings))
    return FaceEnrollResult(
        filename=filename, success=True, face_id=employee_face.id, warnings=warnings
    )


def delete_face(db: Session, face: EmployeeFace, store: EmbeddingStore) -> None:
    db.delete(face)
    db.commit()
    store.refresh(db)
    logger.info("Удалено лицо #%d", face.id)


def get_face(db: Session, employee_id: int, face_id: int) -> Optional[EmployeeFace]:
    return db.query(EmployeeFace).filter(
        EmployeeFace.id == face_id, EmployeeFace.employee_id == employee_id
    ).first()
