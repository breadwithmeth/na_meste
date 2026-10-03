"""Хранилище эмбеддингов сотрудников: загрузка из БД в память, косинусный поиск.

Все эмбеддинги активных сотрудников держатся одной матрицей — сравнение лица
со всеми сотрудниками это одно матричное умножение.
"""
import logging
import threading
from typing import Optional

import numpy as np
from sqlalchemy.orm import Session

from app.database.models import Employee, EmployeeFace

logger = logging.getLogger("app.ai.face_recognition")


class EmbeddingStore:
    def __init__(self):
        self._lock = threading.Lock()
        self._matrix: Optional[np.ndarray] = None   # (N, 512), L2-нормированные
        self._owners: list[int] = []                # employee_id каждой строки
        self._names: dict[int, str] = {}            # employee_id → имя

    def refresh(self, session: Session) -> int:
        """Перезагружает эмбеддинги активных сотрудников из БД."""
        rows = session.query(EmployeeFace.employee_id, EmployeeFace.embedding).join(
            Employee, EmployeeFace.employee_id == Employee.id
        ).filter(Employee.active.is_(True)).all()
        names = {e.id: e.name for e in session.query(Employee).all()}

        vectors, owners = [], []
        for employee_id, blob in rows:
            vec = np.frombuffer(blob, dtype=np.float32)
            if vec.size == 0:
                continue
            norm = np.linalg.norm(vec)
            if norm == 0:
                continue
            vectors.append(vec / norm)
            owners.append(employee_id)

        with self._lock:
            self._matrix = np.stack(vectors) if vectors else None
            self._owners = owners
            self._names = names
        logger.info("Эмбеддинги перезагружены: %d лиц, %d сотрудников", len(vectors), len(names))
        return len(vectors)

    def best_match(self, embedding: np.ndarray) -> tuple[Optional[int], float]:
        """Максимальное косинусное сходство с лицом сотрудника.

        Возвращает (employee_id | None, similarity). None — сотрудников нет.
        """
        vec = np.asarray(embedding, dtype=np.float32)
        norm = np.linalg.norm(vec)
        if norm == 0:
            return None, 0.0
        vec = vec / norm

        with self._lock:
            matrix, owners = self._matrix, list(self._owners)
        if matrix is None or not owners:
            return None, 0.0

        sims = matrix @ vec
        best = int(np.argmax(sims))
        return owners[best], float(sims[best])

    def employee_name(self, employee_id: Optional[int]) -> Optional[str]:
        if employee_id is None:
            return None
        with self._lock:
            return self._names.get(employee_id)
