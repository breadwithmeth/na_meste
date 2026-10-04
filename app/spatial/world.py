"""Мировая модель: снапшот пространственных данных из БД.

Строится при старте и перезагружается целиком после CRUD через API
(reload_world у живого инстанса в AI-воркере). Снапшот неизменяем после
построения, замена ссылки атомарна — блокировки на чтение не нужны.
Модель ничего не знает о трекинге — только геометрия и калибровки.
"""
import json
import threading
from dataclasses import dataclass
from typing import Optional

import numpy as np

from app.spatial.geometry import coverage_from_homography
from app.spatial.navigation import EdgeInfo, NavigationGraph, NodeInfo

# типы объектов, хранящихся как отрезок (остальные — полигоны)
SEGMENT_TYPES = ("wall", "door", "stairs", "elevator")


@dataclass(frozen=True)
class FloorInfo:
    id: int
    name: str
    z: float
    floorplan_scale: Optional[float]       # метров на пиксель подложки
    floorplan_origin: Optional[tuple[float, float]]


@dataclass(frozen=True)
class FeatureInfo:
    id: int
    floor_id: int
    ftype: str
    name: Optional[str]
    geometry: dict                         # распарсенный JSON


@dataclass(frozen=True)
class CalibrationInfo:
    camera_id: int
    homography: np.ndarray                 # 3×3 pixel → world
    resolution: tuple[int, int]
    reprojection_error: float
    points: list[dict]


@dataclass(frozen=True)
class CameraSetupInfo:
    camera_id: int
    floor_id: Optional[int]
    position: tuple[float, float, float]
    rotation: tuple[float, float, float]   # yaw/pitch/roll, градусы
    fov: tuple[float, float]               # горизонтальный/вертикальный
    coverage_polygon: Optional[list[tuple[float, float]]]  # ручной полигон


def _parse_origin(raw: Optional[str]) -> Optional[tuple[float, float]]:
    if not raw:
        return None
    try:
        x, y = json.loads(raw)
        return (float(x), float(y))
    except (ValueError, TypeError):
        return None


class WorldModel:
    """Только чтение; после reload() создаётся новый снапшот."""

    def __init__(self, floors, features, setups, calibrations, nav):
        self.floors: dict[int, FloorInfo] = floors
        self.features: dict[int, list[FeatureInfo]] = features   # по этажам
        self.setups: dict[int, CameraSetupInfo] = setups         # по камерам
        self.calibrations: dict[int, CalibrationInfo] = calibrations
        self.nav: NavigationGraph = nav
        self.version: int = 1

    # ----------------------------------------------------------- фабрика

    @classmethod
    def from_session(cls, db, max_speed: float = 2.0) -> "WorldModel":
        """Построить снапшот из открытой SQLAlchemy-сессии."""
        from app.database.models import (
            SpatialCalibration, SpatialCameraSetup, SpatialEdge,
            SpatialFeature, SpatialFloor, SpatialNode,
        )

        floors: dict[int, FloorInfo] = {}
        for f in db.query(SpatialFloor).all():
            floors[f.id] = FloorInfo(
                id=f.id, name=f.name, z=float(f.z),
                floorplan_scale=float(f.floorplan_scale)
                if f.floorplan_scale else None,
                floorplan_origin=_parse_origin(f.floorplan_origin),
            )

        features: dict[int, list[FeatureInfo]] = {fid: [] for fid in floors}
        for feat in db.query(SpatialFeature).all():
            try:
                geometry = json.loads(feat.geometry)
            except (ValueError, TypeError):
                continue
            if feat.floor_id in features:
                features[feat.floor_id].append(FeatureInfo(
                    id=feat.id, floor_id=feat.floor_id, ftype=feat.ftype,
                    name=feat.name, geometry=geometry,
                ))

        setups: dict[int, CameraSetupInfo] = {}
        for s in db.query(SpatialCameraSetup).all():
            setups[s.camera_id] = CameraSetupInfo(
                camera_id=s.camera_id, floor_id=s.floor_id,
                position=(float(s.pos_x), float(s.pos_y), float(s.pos_z)),
                rotation=(float(s.yaw), float(s.pitch), float(s.roll)),
                fov=(float(s.fov_h), float(s.fov_v)),
                coverage_polygon=_parse_polygon(s.coverage_polygon),
            )

        calibrations: dict[int, CalibrationInfo] = {}
        for c in db.query(SpatialCalibration).all():
            try:
                H = np.array(json.loads(c.homography), dtype=np.float64)
                points = json.loads(c.calibration_points)
            except (ValueError, TypeError):
                continue
            if H.shape != (3, 3):
                continue
            calibrations[c.camera_id] = CalibrationInfo(
                camera_id=c.camera_id, homography=H,
                resolution=(int(c.resolution_w), int(c.resolution_h)),
                reprojection_error=float(c.reprojection_error),
                points=points,
            )

        nodes = [NodeInfo(id=n.id, floor_id=n.floor_id, ntype=n.ntype,
                          name=n.name, x=float(n.pos_x), y=float(n.pos_y))
                 for n in db.query(SpatialNode).all()]
        edges = [EdgeInfo(id=e.id, from_id=e.from_node, to_id=e.to_node,
                          distance=float(e.distance),
                          min_time=e.min_time, max_time=e.max_time)
                 for e in db.query(SpatialEdge).all()]
        nav = NavigationGraph(nodes, edges, max_speed=max_speed)

        return cls(floors, features, setups, calibrations, nav)

    # --------------------------------------------------------- запросы

    def floor_of_camera(self, camera_id: int) -> Optional[int]:
        setup = self.setups.get(camera_id)
        return setup.floor_id if setup else None

    def calibration(self, camera_id: int) -> Optional[CalibrationInfo]:
        return self.calibrations.get(camera_id)

    def coverage(self, camera_id: int) -> Optional[list[tuple[float, float]]]:
        """Зона видимости камеры: ручной полигон, иначе автоматически из
        гомографии (проекция рамки кадра на пол)."""
        setup = self.setups.get(camera_id)
        if setup is not None and setup.coverage_polygon:
            return setup.coverage_polygon
        calib = self.calibrations.get(camera_id)
        if calib is None:
            return None
        return coverage_from_homography(
            calib.homography, calib.resolution[0], calib.resolution[1])

    def walls(self, floor_id: int) -> list[tuple[tuple[float, float],
                                                 tuple[float, float]]]:
        """Отрезки стен этажа (для штрафа «прямая сквозь стену»)."""
        out = []
        for f in self.features.get(floor_id, []):
            if f.ftype != "wall":
                continue
            start, end = f.geometry.get("start"), f.geometry.get("end")
            if start and end:
                out.append(((float(start[0]), float(start[1])),
                            (float(end[0]), float(end[1]))))
        return out

    def features_of(self, floor_id: int,
                    ftype: Optional[str] = None) -> list[FeatureInfo]:
        items = self.features.get(floor_id, [])
        return [f for f in items if ftype is None or f.ftype == ftype]


def _parse_polygon(raw: Optional[str]) -> Optional[list[tuple[float, float]]]:
    if not raw:
        return None
    try:
        data = json.loads(raw)
        return [(float(p[0]), float(p[1])) for p in data]
    except (ValueError, TypeError, IndexError):
        return None


class WorldHolder:
    """Потокобезопасный держатель текущего снапшота мира (замена после CRUD)."""

    def __init__(self, world: WorldModel):
        self._lock = threading.RLock()
        self._world = world

    @property
    def current(self) -> WorldModel:
        with self._lock:
            return self._world

    def replace(self, world: WorldModel) -> None:
        with self._lock:
            world.version = self._world.version + 1
            self._world = world
