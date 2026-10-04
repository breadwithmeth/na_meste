"""REST API 2.5D Spatial World Model (ТЗ §17).

Чтение мира:      GET /world, /floors, /cameras, /coverage
Калибровка:       GET|POST|DELETE /cameras/{id}/calibration
Живые данные:     GET /live, /trajectories/{gid}, /observations/{gid},
                  /prediction/{gid}, /debug/matches
Редактор:         CRUD этажей (+floorplan), фич (стены/двери/зоны),
                  сетапов камер, узлов и рёбер навигационного графа

API работает всегда (мир можно настроить до включения SPATIAL_MODEL_ENABLED);
живые данные (/live, /prediction) — только при запущенном AI-воркере
с включённой моделью. После мутаций живой инстанс в воркере перезагружает
мир (reload_world).
"""
import json
import logging
from typing import Optional

import cv2
import numpy as np
from fastapi import APIRouter, Depends, HTTPException, Query, Request, UploadFile
from fastapi.responses import Response
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.database.database import get_db
from app.database.models import (
    Camera, SpatialCalibration, SpatialCameraSetup, SpatialEdge,
    SpatialFeature, SpatialFloor, SpatialNode, SpatialObservation,
)
from app.rtsp.manager import ReaderManager
from app.spatial.geometry import compute_homography, pixel_to_world
from app.spatial.markers import detect_markers, generate_pages_pdf, generate_sheet
from app.spatial.world import WorldModel

logger = logging.getLogger("app.api.spatial")

router = APIRouter(prefix="/api/spatial", tags=["spatial"])

FLOORPLAN_MAX_BYTES = 10 * 1024 * 1024
SEGMENT_TYPES = ("wall", "door", "stairs", "elevator")
POLYGON_TYPES = ("zone", "restricted_zone", "entrance", "exit",
                 "corridor", "room")
NODE_TYPES = ("corridor", "room", "door", "stairs", "elevator",
              "entrance", "exit", "restricted_zone")
FLOOR_TRANSITION_TYPES = ("stairs", "elevator")


# ------------------------------------------------------------------ схемы

class FloorCreate(BaseModel):
    name: str = Field(min_length=1, max_length=100)
    z: float = 0.0


class FloorUpdate(BaseModel):
    name: Optional[str] = Field(default=None, min_length=1, max_length=100)
    z: Optional[float] = None


class FloorplanTransform(BaseModel):
    scale: float = Field(gt=0)             # метров на пиксель подложки
    origin: list[float]                    # мировые координаты угла [x, y]


class FeatureCreate(BaseModel):
    type: str
    name: Optional[str] = None
    geometry: dict


class FeatureUpdate(BaseModel):
    name: Optional[str] = None
    geometry: Optional[dict] = None


class CameraSetupIn(BaseModel):
    floor_id: Optional[int] = None
    position: dict = Field(
        default_factory=lambda: {"x": 0.0, "y": 0.0, "z": 3.0})
    rotation: dict = Field(
        default_factory=lambda: {"yaw": 0.0, "pitch": 0.0, "roll": 0.0})
    fov: dict = Field(
        default_factory=lambda: {"horizontal": 90.0, "vertical": 55.0})
    coverage_polygon: Optional[list[list[float]]] = None   # null = авто


class CalibrationPoint(BaseModel):
    pixel: list[float]
    world: list[float]


class CalibrationIn(BaseModel):
    points: list[CalibrationPoint]
    resolution: list[int]                   # [width, height] потока


class MarkerWorldPoint(BaseModel):
    marker_id: int
    world: list[float]


class AutoCalibrationIn(BaseModel):
    """Авто-калибровка по ArUco-маркерам: ID маркера → мировая координата
    его центра. Пиксели детектор берёт из живого кадра сам."""
    points: list[MarkerWorldPoint]


class NodeCreate(BaseModel):
    floor_id: int
    type: str
    name: Optional[str] = None
    position: list[float]                   # [x, y]


class NodeUpdate(BaseModel):
    name: Optional[str] = None
    position: Optional[list[float]] = None


class EdgeCreate(BaseModel):
    from_node: int
    to_node: int
    distance: Optional[float] = None        # null = евклидово между узлами
    min_time: Optional[float] = None
    max_time: Optional[float] = None


# ---------------------------------------------------------------- helpers

def _spatial(request: Request):
    """Живой SpatialWorldModel из AI-воркера (None — выключен)."""
    worker = getattr(request.app.state, "ai_worker", None)
    return getattr(worker, "spatial_model", None) if worker else None


def _touch_spatial(request: Request) -> None:
    """Перезагрузить мир в воркере после CRUD."""
    spatial = _spatial(request)
    if spatial is not None:
        try:
            spatial.reload_world()
        except Exception:
            logger.exception("Spatial: не перезагрузить мир в воркере")


def _floor_or_404(db: Session, floor_id: int) -> SpatialFloor:
    floor = db.get(SpatialFloor, floor_id)
    if floor is None:
        raise HTTPException(404, "Этаж не найден")
    return floor


def _feature_or_404(db: Session, feature_id: int) -> SpatialFeature:
    feature = db.get(SpatialFeature, feature_id)
    if feature is None:
        raise HTTPException(404, "Объект не найден")
    return feature


def _camera_or_404(db: Session, camera_id: int) -> Camera:
    camera = db.get(Camera, camera_id)
    if camera is None:
        raise HTTPException(404, "Камера не найдена")
    return camera


def _get_manager(request: Request) -> ReaderManager:
    return request.app.state.manager


def _grab_frame(request: Request, camera_id: int):
    """Живой кадр из буфера ридера; (503, None) если камера не отдаёт."""
    manager = _get_manager(request)
    reader = manager.get(camera_id)
    if reader is None or not reader.is_alive():
        raise HTTPException(503, "Камера не подключена")
    item = reader.buffer.latest()
    if item is None:
        raise HTTPException(503, "Кадр недоступен: камера не отдаёт поток")
    return item[1]


def _validate_geometry(ftype: str, geometry: dict) -> dict:
    start, end = geometry.get("start"), geometry.get("end")
    points = geometry.get("points")
    if ftype in SEGMENT_TYPES:
        if not start or not end or len(start) < 2 or len(end) < 2:
            raise HTTPException(400, "Для отрезка нужны start и end [x, y]")
        out = {
            "start": [float(start[0]), float(start[1])],
            "end": [float(end[0]), float(end[1])],
        }
        height = geometry.get("height")
        if height is not None:
            out["height"] = float(height)     # высота стен/проёмов, м
        return out
    if ftype in POLYGON_TYPES:
        if not points or len(points) < 3:
            raise HTTPException(400, "Для полигона нужно минимум 3 точки")
        return {"points": [[float(p[0]), float(p[1])] for p in points]}
    raise HTTPException(400, f"Неизвестный тип объекта: {ftype}")


def _world_or_build(db: Session, request: Request) -> WorldModel:
    """Живая модель из воркера или свежий снапшот из БД (когда воркер
    выключен — редактор всё равно должен работать)."""
    spatial = _spatial(request)
    if spatial is not None:
        return spatial.world
    return WorldModel.from_session(db)


# ------------------------------------------------------------------- мир

@router.get("/world")
def api_world(request: Request, db: Session = Depends(get_db)):
    """Полный мир: этажи с объектами, камеры (сетап + калибровка),
    навигационный граф."""
    world = _world_or_build(db, request)
    camera_names = {c.id: c.name for c in db.query(Camera).all()}
    floors_out = []
    for floor in world.floors.values():
        floors_out.append({
            "id": floor.id,
            "name": floor.name,
            "z": floor.z,
            "floorplan_scale": floor.floorplan_scale,
            "floorplan_origin": list(floor.floorplan_origin)
            if floor.floorplan_origin else None,
            "features": [
                {"id": f.id, "type": f.ftype, "name": f.name,
                 "geometry": f.geometry}
                for f in world.features_of(floor.id)
            ],
        })
    cameras_out = []
    for camera_id, setup in world.setups.items():
        calib = world.calibrations.get(camera_id)
        coverage = world.coverage(camera_id)
        cameras_out.append({
            "camera_id": camera_id,
            "camera_name": camera_names.get(camera_id, f"камера #{camera_id}"),
            "floor_id": setup.floor_id,
            "position": {"x": setup.position[0], "y": setup.position[1],
                         "z": setup.position[2]},
            "rotation": {"yaw": setup.rotation[0], "pitch": setup.rotation[1],
                         "roll": setup.rotation[2]},
            "fov": {"horizontal": setup.fov[0], "vertical": setup.fov[1]},
            "coverage_polygon": [[round(p[0], 2), round(p[1], 2)]
                                 for p in coverage] if coverage else None,
            "coverage_source": ("manual" if setup.coverage_polygon
                                else "auto") if coverage else None,
            "calibrated": calib is not None,
            "resolution": list(calib.resolution) if calib else None,
            "reprojection_error": round(calib.reprojection_error, 3)
            if calib else None,
        })
    # камеры без сетапа — чтобы редактор мог их разместить
    with_setup = {c["camera_id"] for c in cameras_out}
    for camera_id, name in camera_names.items():
        if camera_id not in with_setup:
            cameras_out.append({
                "camera_id": camera_id, "camera_name": name,
                "floor_id": None, "position": None, "rotation": None,
                "fov": None, "coverage_polygon": None,
                "coverage_source": None, "calibrated": False,
                "resolution": None, "reprojection_error": None,
            })
    nodes_out = [
        {"id": n.id, "floor_id": n.floor_id, "type": n.ntype, "name": n.name,
         "position": [n.x, n.y]}
        for n in world.nav.nodes()
    ]
    node_by_id = {n["id"]: n for n in nodes_out}
    edges_out = [
        {"id": e.id, "from_node": e.from_id, "to_node": e.to_id,
         "distance": round(e.distance, 2), "min_time": e.min_time,
         "max_time": e.max_time}
        for e in world.nav.edges()
    ]
    return {
        "floors": floors_out,
        "cameras": cameras_out,
        "nodes": nodes_out,
        "edges": [
            {**e,
             "from_floor": node_by_id[e["from_node"]]["floor_id"]
             if e["from_node"] in node_by_id else None,
             "to_floor": node_by_id[e["to_node"]]["floor_id"]
             if e["to_node"] in node_by_id else None}
            for e in edges_out
        ],
        "spatial_model_enabled": _spatial(request) is not None,
    }


@router.get("/floors")
def api_list_floors(db: Session = Depends(get_db)):
    return [
        {"id": f.id, "name": f.name, "z": f.z,
         "has_floorplan": f.floorplan is not None,
         "floorplan_scale": f.floorplan_scale,
         "floorplan_origin": json.loads(f.floorplan_origin)
         if f.floorplan_origin else None}
        for f in db.query(SpatialFloor).order_by(SpatialFloor.id).all()
    ]


@router.post("/floors", status_code=201)
def api_create_floor(payload: FloorCreate, request: Request,
                     db: Session = Depends(get_db)):
    floor = SpatialFloor(name=payload.name, z=payload.z)
    db.add(floor)
    db.commit()
    db.refresh(floor)
    _touch_spatial(request)
    return {"id": floor.id, "name": floor.name, "z": floor.z}


@router.put("/floors/{floor_id}")
def api_update_floor(floor_id: int, payload: FloorUpdate, request: Request,
                     db: Session = Depends(get_db)):
    floor = _floor_or_404(db, floor_id)
    if payload.name is not None:
        floor.name = payload.name
    if payload.z is not None:
        floor.z = payload.z
    db.commit()
    _touch_spatial(request)
    return {"id": floor.id, "name": floor.name, "z": floor.z}


@router.delete("/floors/{floor_id}", status_code=204)
def api_delete_floor(floor_id: int, request: Request,
                     db: Session = Depends(get_db)):
    floor = _floor_or_404(db, floor_id)
    # удаляем содержимое явно (не полагаясь на FK-каскады SQLite)
    db.query(SpatialFeature).filter_by(floor_id=floor_id).delete()
    node_ids = [n.id for n in db.query(SpatialNode)
                .filter_by(floor_id=floor_id).all()]
    if node_ids:
        db.query(SpatialEdge).filter(
            SpatialEdge.from_node.in_(node_ids)
            | SpatialEdge.to_node.in_(node_ids)).delete(synchronize_session=False)
        db.query(SpatialNode).filter_by(floor_id=floor_id).delete()
    for setup in db.query(SpatialCameraSetup).filter_by(floor_id=floor_id).all():
        setup.floor_id = None
    db.delete(floor)
    db.commit()
    _touch_spatial(request)
    return None


# -------------------------------------------------------------- floorplan

@router.post("/floors/{floor_id}/floorplan")
async def api_upload_floorplan(floor_id: int, request: Request,
                               file: UploadFile, db: Session = Depends(get_db)):
    """Загрузить планировку этажа (изображение). Перекодируется в JPEG."""
    floor = _floor_or_404(db, floor_id)
    raw = await file.read()
    if len(raw) > FLOORPLAN_MAX_BYTES:
        raise HTTPException(413, "Файл больше 10 МБ")
    img = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(400, "Не удалось прочитать изображение")
    ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    if not ok:
        raise HTTPException(400, "Не удалось перекодировать изображение")
    floor.floorplan = buf.tobytes()
    db.commit()
    _touch_spatial(request)
    return {"width": int(img.shape[1]), "height": int(img.shape[0])}


@router.get("/floors/{floor_id}/floorplan")
def api_get_floorplan(floor_id: int, db: Session = Depends(get_db)):
    floor = _floor_or_404(db, floor_id)
    if not floor.floorplan:
        raise HTTPException(404, "Планировка не загружена")
    return Response(content=floor.floorplan, media_type="image/jpeg",
                    headers={"Cache-Control": "no-cache"})


@router.delete("/floors/{floor_id}/floorplan", status_code=204)
def api_delete_floorplan(floor_id: int, request: Request,
                         db: Session = Depends(get_db)):
    floor = _floor_or_404(db, floor_id)
    floor.floorplan = None
    floor.floorplan_scale = None
    floor.floorplan_origin = None
    db.commit()
    _touch_spatial(request)
    return None


@router.put("/floors/{floor_id}/floorplan/transform")
def api_floorplan_transform(floor_id: int, payload: FloorplanTransform,
                            request: Request, db: Session = Depends(get_db)):
    """Масштаб (м/пиксель) и привязка (мировые координаты угла подложки)."""
    floor = _floor_or_404(db, floor_id)
    if not floor.floorplan:
        raise HTTPException(400, "Сначала загрузите планировку")
    floor.floorplan_scale = payload.scale
    floor.floorplan_origin = json.dumps(payload.origin)
    db.commit()
    _touch_spatial(request)
    return {"scale": floor.floorplan_scale,
            "origin": payload.origin}


# ---------------------------------------------------------------- фичи

@router.get("/floors/{floor_id}/features")
def api_list_features(floor_id: int, db: Session = Depends(get_db)):
    _floor_or_404(db, floor_id)
    return [
        {"id": f.id, "type": f.ftype, "name": f.name,
         "geometry": json.loads(f.geometry)}
        for f in db.query(SpatialFeature).filter_by(floor_id=floor_id).all()
    ]


@router.post("/floors/{floor_id}/features", status_code=201)
def api_create_feature(floor_id: int, payload: FeatureCreate,
                       request: Request, db: Session = Depends(get_db)):
    _floor_or_404(db, floor_id)
    geometry = _validate_geometry(payload.type, payload.geometry)
    feature = SpatialFeature(floor_id=floor_id, ftype=payload.type,
                             name=payload.name, geometry=json.dumps(geometry))
    db.add(feature)
    db.commit()
    db.refresh(feature)
    _touch_spatial(request)
    return {"id": feature.id, "type": feature.ftype, "name": feature.name,
            "geometry": geometry}


@router.put("/features/{feature_id}")
def api_update_feature(feature_id: int, payload: FeatureUpdate,
                       request: Request, db: Session = Depends(get_db)):
    feature = _feature_or_404(db, feature_id)
    if payload.name is not None:
        feature.name = payload.name
    if payload.geometry is not None:
        feature.geometry = json.dumps(
            _validate_geometry(feature.ftype, payload.geometry))
    db.commit()
    _touch_spatial(request)
    return {"id": feature.id, "type": feature.ftype, "name": feature.name,
            "geometry": json.loads(feature.geometry)}


@router.delete("/features/{feature_id}", status_code=204)
def api_delete_feature(feature_id: int, request: Request,
                       db: Session = Depends(get_db)):
    feature = _feature_or_404(db, feature_id)
    db.delete(feature)
    db.commit()
    _touch_spatial(request)
    return None


# --------------------------------------------------------------- камеры

@router.get("/cameras")
def api_list_spatial_cameras(request: Request, db: Session = Depends(get_db)):
    """Сетапы всех камер + статус калибровки (сетап — upsert из редактора)."""
    world = _world_or_build(db, request)
    names = {c.id: c.name for c in db.query(Camera).all()}
    out = []
    for camera_id, name in names.items():
        setup = world.setups.get(camera_id)
        calib = world.calibrations.get(camera_id)
        coverage = world.coverage(camera_id)
        out.append({
            "camera_id": camera_id,
            "camera_name": name,
            "floor_id": setup.floor_id if setup else None,
            "position": {"x": setup.position[0], "y": setup.position[1],
                         "z": setup.position[2]} if setup else None,
            "calibrated": calib is not None,
            "resolution": list(calib.resolution) if calib else None,
            "reprojection_error": round(calib.reprojection_error, 3)
            if calib else None,
            "coverage_polygon": [[round(p[0], 2), round(p[1], 2)]
                                 for p in coverage] if coverage else None,
        })
    return out


@router.put("/cameras/{camera_id}")
def api_update_camera_setup(camera_id: int, payload: CameraSetupIn,
                            request: Request, db: Session = Depends(get_db)):
    """Сохранить пространственное положение камеры (upsert)."""
    _camera_or_404(db, camera_id)
    if payload.floor_id is not None:
        _floor_or_404(db, payload.floor_id)
    setup = db.get(SpatialCameraSetup, camera_id)
    values = dict(
        floor_id=payload.floor_id,
        pos_x=payload.position["x"], pos_y=payload.position["y"],
        pos_z=payload.position["z"],
        yaw=payload.rotation["yaw"], pitch=payload.rotation["pitch"],
        roll=payload.rotation["roll"],
        fov_h=payload.fov["horizontal"], fov_v=payload.fov["vertical"],
        coverage_polygon=json.dumps(payload.coverage_polygon)
        if payload.coverage_polygon else None,
    )
    if setup is None:
        setup = SpatialCameraSetup(camera_id=camera_id, **values)
        db.add(setup)
    else:
        for key, value in values.items():
            setattr(setup, key, value)
    db.commit()
    _touch_spatial(request)
    return {"camera_id": camera_id, **{k: v for k, v in values.items()
                                        if k != "coverage_polygon"},
            "coverage_polygon": payload.coverage_polygon}


# ------------------------------------------------------------ калибровка

@router.get("/cameras/{camera_id}/calibration")
def api_get_calibration(camera_id: int, db: Session = Depends(get_db)):
    """Текущая калибровка камеры (гомография + точки + ошибка)."""
    _camera_or_404(db, camera_id)
    calib = db.get(SpatialCalibration, camera_id)
    if calib is None:
        return {"camera_id": camera_id, "calibrated": False}
    return {
        "camera_id": camera_id,
        "calibrated": True,
        "homography": json.loads(calib.homography),
        "calibration_points": json.loads(calib.calibration_points),
        "resolution": [calib.resolution_w, calib.resolution_h],
        "reprojection_error": round(calib.reprojection_error, 3),
        "updated_at": calib.updated_at.isoformat(timespec="seconds"),
    }


@router.post("/cameras/{camera_id}/calibration")
def api_calibrate_camera(camera_id: int, payload: CalibrationIn,
                         request: Request, db: Session = Depends(get_db)):
    """Вычислить и сохранить гомографию pixel → world по ≥4 точкам пола.

    Перезапись = перекалибровка. Возвращает гомографию, ошибку репроекции
    и автоматически посчитанную зону видимости.
    """
    _camera_or_404(db, camera_id)
    if len(payload.resolution) != 2 or payload.resolution[0] <= 0:
        raise HTTPException(400, "resolution = [width, height]")
    if len(payload.points) < 4:
        raise HTTPException(400, "Нужно минимум 4 пары точек")
    points = [{"pixel": list(p.pixel), "world": list(p.world)}
              for p in payload.points]
    try:
        H, error = compute_homography(points)
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    if error > 1.0:
        # не отказываем, но предупреждаем — точки противоречивы
        logger.warning("Калибровка camera=%d: большая ошибка %.2f м",
                       camera_id, error)
    calib = db.get(SpatialCalibration, camera_id)
    values = dict(
        homography=json.dumps([[float(v) for v in row] for row in H]),
        calibration_points=json.dumps(points),
        resolution_w=int(payload.resolution[0]),
        resolution_h=int(payload.resolution[1]),
        reprojection_error=float(error),
    )
    if calib is None:
        calib = SpatialCalibration(camera_id=camera_id, **values)
        db.add(calib)
    else:
        for key, value in values.items():
            setattr(calib, key, value)
    db.commit()
    _touch_spatial(request)
    # авто-зона видимости для ответа (сохранена неявно — считается из H)
    from app.spatial.geometry import coverage_from_homography
    coverage = coverage_from_homography(
        H, int(payload.resolution[0]), int(payload.resolution[1]))
    return {
        "camera_id": camera_id,
        "calibrated": True,
        "homography": [[float(v) for v in row] for row in H],
        "reprojection_error": round(float(error), 3),
        "coverage_polygon": [[round(p[0], 2), round(p[1], 2)]
                             for p in coverage] if coverage else None,
    }


@router.delete("/cameras/{camera_id}/calibration", status_code=204)
def api_delete_calibration(camera_id: int, request: Request,
                           db: Session = Depends(get_db)):
    _camera_or_404(db, camera_id)
    calib = db.get(SpatialCalibration, camera_id)
    if calib is not None:
        db.delete(calib)
        db.commit()
    _touch_spatial(request)
    return None


# ------------------------------------------------- калибровка по маркерам

@router.get("/markers/sheet")
def api_markers_sheet(count: int = Query(6, ge=1, le=50),
                      marker_cm: float = Query(10.0, ge=5.0, le=18.0),
                      dpi: int = Query(150, ge=96, le=300),
                      one_per_page: bool = False):
    """Лист ArUco-маркеров. По умолчанию — сетка на одном листе A4 (PNG);
    one_per_page=true — по одному крупному маркеру на страницу (PDF).
    Печать в масштабе 100%, контрольный отрезок 10 см на каждом листе."""
    try:
        if one_per_page:
            pdf = generate_pages_pdf(count=count, marker_cm=marker_cm,
                                     dpi=dpi)
            return Response(
                content=pdf, media_type="application/pdf",
                headers={"Content-Disposition":
                         'inline; filename="aruco_markers.pdf"'})
        png = generate_sheet(count=count, marker_cm=marker_cm, dpi=dpi)
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    return Response(content=png, media_type="image/png",
                    headers={"Content-Disposition":
                             'inline; filename="aruco_markers.png"'})


@router.get("/cameras/{camera_id}/markers")
def api_detect_markers(camera_id: int, request: Request,
                       db: Session = Depends(get_db)):
    """Найти ArUco-маркеры в текущем кадре камеры (id, центр, размер)."""
    _camera_or_404(db, camera_id)
    frame = _grab_frame(request, camera_id)
    markers = detect_markers(frame)
    return {
        "camera_id": camera_id,
        "resolution": [int(frame.shape[1]), int(frame.shape[0])],
        "markers": [
            {"marker_id": m.marker_id, "pixel": [round(m.pixel[0], 1),
                                                 round(m.pixel[1], 1)],
             "size_px": round(m.size_px, 1)}
            for m in markers
        ],
    }


@router.post("/cameras/{camera_id}/markers/measure")
def api_measure_markers(camera_id: int, request: Request,
                        db: Session = Depends(get_db)):
    """Цепочная калибровка: определить мировые координаты маркеров через
    УЖЕ откалиброванную камеру (гомография проецирует центры маркеров
    на пол). Результат заполняет таблицу для калибровки соседних камер,
    видящих те же маркеры, — без рулетки."""
    _camera_or_404(db, camera_id)
    calib = db.get(SpatialCalibration, camera_id)
    if calib is None:
        raise HTTPException(400, "Камера ещё не откалибрована")
    frame = _grab_frame(request, camera_id)
    H = np.array(json.loads(calib.homography), dtype=np.float64)
    out = []
    for m in detect_markers(frame):
        world = pixel_to_world(H, m.pixel[0], m.pixel[1])
        if world is not None:
            out.append({"marker_id": m.marker_id,
                        "world": [round(world[0], 2), round(world[1], 2)]})
    return {"camera_id": camera_id, "markers": out}


@router.post("/cameras/{camera_id}/calibration/auto")
def api_auto_calibrate(camera_id: int, payload: AutoCalibrationIn,
                       request: Request, db: Session = Depends(get_db)):
    """Авто-калибровка: детект ArUco-маркеров в живом кадре + таблица
    «ID → мировая координата центра» → гомография. Перезапись =
    перекалибровка. Разрешение берётся из самого кадра."""
    _camera_or_404(db, camera_id)
    if len(payload.points) < 4:
        raise HTTPException(400, "Нужны координаты минимум 4 маркеров")
    frame = _grab_frame(request, camera_id)
    detected = {m.marker_id: m for m in detect_markers(frame)}
    pairs, missing = [], []
    for p in payload.points:
        marker = detected.get(p.marker_id)
        if marker is None:
            missing.append(p.marker_id)
            continue
        pairs.append({"pixel": [marker.pixel[0], marker.pixel[1]],
                      "world": list(p.world)})
    if len(pairs) < 4:
        found = sorted(detected.keys())
        raise HTTPException(400, (
            f"В кадре сопоставлено только {len(pairs)} маркера(ов) — "
            f"нужно ≥4. Не найдены: {missing or 'нет'}; "
            f"детектор видел ID {found or 'ничего'}. "
            f"Проверьте, что маркеры видны, плоско наклеены и не бликуют"))
    try:
        H, error = compute_homography(pairs)
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    values = dict(
        homography=json.dumps([[float(v) for v in row] for row in H]),
        calibration_points=json.dumps(pairs),
        resolution_w=int(frame.shape[1]),
        resolution_h=int(frame.shape[0]),
        reprojection_error=float(error),
    )
    calib = db.get(SpatialCalibration, camera_id)
    if calib is None:
        calib = SpatialCalibration(camera_id=camera_id, **values)
        db.add(calib)
    else:
        for key, value in values.items():
            setattr(calib, key, value)
    db.commit()
    _touch_spatial(request)
    from app.spatial.geometry import coverage_from_homography
    coverage = coverage_from_homography(H, int(frame.shape[1]),
                                        int(frame.shape[0]))
    return {
        "camera_id": camera_id,
        "calibrated": True,
        "homography": [[float(v) for v in row] for row in H],
        "reprojection_error": round(float(error), 3),
        "resolution": [int(frame.shape[1]), int(frame.shape[0])],
        "used_markers": sorted(p.marker_id for p in payload.points
                               if p.marker_id not in missing),
        "missing_markers": missing,
        "markers_in_frame": sorted(detected.keys()),
        "coverage_polygon": [[round(p[0], 2), round(p[1], 2)]
                             for p in coverage] if coverage else None,
    }


# ----------------------------------------------------------- навигация

@router.get("/nodes")
def api_list_nodes(db: Session = Depends(get_db)):
    return [
        {"id": n.id, "floor_id": n.floor_id, "type": n.ntype, "name": n.name,
         "position": [n.pos_x, n.pos_y]}
        for n in db.query(SpatialNode).order_by(SpatialNode.id).all()
    ]


@router.post("/nodes", status_code=201)
def api_create_node(payload: NodeCreate, request: Request,
                    db: Session = Depends(get_db)):
    _floor_or_404(db, payload.floor_id)
    if payload.type not in NODE_TYPES:
        raise HTTPException(400, f"Тип узла: один из {NODE_TYPES}")
    if len(payload.position) < 2:
        raise HTTPException(400, "position = [x, y]")
    node = SpatialNode(floor_id=payload.floor_id, ntype=payload.type,
                       name=payload.name, pos_x=payload.position[0],
                       pos_y=payload.position[1])
    db.add(node)
    db.commit()
    db.refresh(node)
    _touch_spatial(request)
    return {"id": node.id, "floor_id": node.floor_id, "type": node.ntype,
            "name": node.name, "position": [node.pos_x, node.pos_y]}


@router.put("/nodes/{node_id}")
def api_update_node(node_id: int, payload: NodeUpdate, request: Request,
                    db: Session = Depends(get_db)):
    node = db.get(SpatialNode, node_id)
    if node is None:
        raise HTTPException(404, "Узел не найден")
    if payload.name is not None:
        node.name = payload.name
    if payload.position is not None:
        if len(payload.position) < 2:
            raise HTTPException(400, "position = [x, y]")
        node.pos_x, node.pos_y = payload.position[0], payload.position[1]
    db.commit()
    _touch_spatial(request)
    return {"id": node.id, "floor_id": node.floor_id, "type": node.ntype,
            "name": node.name, "position": [node.pos_x, node.pos_y]}


@router.delete("/nodes/{node_id}", status_code=204)
def api_delete_node(node_id: int, request: Request,
                    db: Session = Depends(get_db)):
    node = db.get(SpatialNode, node_id)
    if node is None:
        raise HTTPException(404, "Узел не найден")
    # связанные рёбра удаляем явно (не полагаясь на FK-каскады SQLite)
    db.query(SpatialEdge).filter(
        (SpatialEdge.from_node == node_id) | (SpatialEdge.to_node == node_id)
    ).delete(synchronize_session=False)
    db.delete(node)
    db.commit()
    _touch_spatial(request)
    return None


@router.get("/edges")
def api_list_edges(db: Session = Depends(get_db)):
    return [
        {"id": e.id, "from_node": e.from_node, "to_node": e.to_node,
         "distance": round(e.distance, 2), "min_time": e.min_time,
         "max_time": e.max_time}
        for e in db.query(SpatialEdge).order_by(SpatialEdge.id).all()
    ]


@router.post("/edges", status_code=201)
def api_create_edge(payload: EdgeCreate, request: Request,
                    db: Session = Depends(get_db)):
    a, b = db.get(SpatialNode, payload.from_node), \
        db.get(SpatialNode, payload.to_node)
    if a is None or b is None:
        raise HTTPException(404, "Узел не найден")
    if a.id == b.id:
        raise HTTPException(400, "Ребро в сам себя не имеет смысла")
    # межэтажный переход — только через stairs/elevator (ТЗ §12)
    if (a.floor_id != b.floor_id
            and a.ntype not in FLOOR_TRANSITION_TYPES
            and b.ntype not in FLOOR_TRANSITION_TYPES):
        raise HTTPException(
            400, "Межэтажное ребро допустимо только через узел stairs/elevator")
    duplicate = db.query(SpatialEdge).filter(
        ((SpatialEdge.from_node == a.id) & (SpatialEdge.to_node == b.id))
        | ((SpatialEdge.from_node == b.id) & (SpatialEdge.to_node == a.id))
    ).first()
    if duplicate is not None:
        raise HTTPException(400, "Ребро между этими узлами уже есть")
    distance = payload.distance
    if distance is None:
        distance = float(np.hypot(a.pos_x - b.pos_x, a.pos_y - b.pos_y))
    edge = SpatialEdge(from_node=a.id, to_node=b.id, distance=distance,
                       min_time=payload.min_time, max_time=payload.max_time)
    db.add(edge)
    db.commit()
    db.refresh(edge)
    _touch_spatial(request)
    return {"id": edge.id, "from_node": edge.from_node,
            "to_node": edge.to_node, "distance": round(edge.distance, 2),
            "min_time": edge.min_time, "max_time": edge.max_time}


@router.delete("/edges/{edge_id}", status_code=204)
def api_delete_edge(edge_id: int, request: Request,
                    db: Session = Depends(get_db)):
    edge = db.get(SpatialEdge, edge_id)
    if edge is None:
        raise HTTPException(404, "Ребро не найдено")
    db.delete(edge)
    db.commit()
    _touch_spatial(request)
    return None


# ----------------------------------------------------- живые данные

@router.get("/live")
def api_live(request: Request):
    """Текущие люди в мировых координатах (для карты; поллинг ~1 с)."""
    spatial = _spatial(request)
    if spatial is None:
        return {"spatial_model_enabled": False, "people": []}
    return {"spatial_model_enabled": True, "people": spatial.live()}


@router.get("/coverage")
def api_coverage(request: Request, db: Session = Depends(get_db)):
    """Зоны видимости камер: где человек может быть виден (и blind spots —
    всё, что вне полигонов)."""
    world = _world_or_build(db, request)
    out = []
    for camera_id in world.setups:
        coverage = world.coverage(camera_id)
        out.append({
            "camera_id": camera_id,
            "floor_id": world.floor_of_camera(camera_id),
            "coverage_polygon": [[round(p[0], 2), round(p[1], 2)]
                                 for p in coverage] if coverage else None,
        })
    return {"cameras": out}


@router.get("/trajectories/{global_id}")
def api_trajectory(global_id: int, request: Request,
                   limit: int = 500, db: Session = Depends(get_db)):
    """История мировой траектории личности (БД + живой хвост)."""
    spatial = _spatial(request)
    if spatial is not None:
        points = spatial.person_history(global_id, limit=limit)
    else:
        points = _observations_from_db(db, global_id, limit)
    return {"global_id": global_id, "points": points}


@router.get("/observations/{global_id}")
def api_observations(global_id: int, limit: int = 200,
                     db: Session = Depends(get_db)):
    """SpatialObservation личности (мировые позиции, скорость)."""
    return _observations_from_db(db, global_id, limit)


def _observations_from_db(db: Session, global_id: int, limit: int) -> list[dict]:
    rows = db.query(SpatialObservation).filter(
        SpatialObservation.global_id == global_id,
    ).order_by(SpatialObservation.created_at.desc()).limit(limit).all()
    return [{
        "time": o.created_at.isoformat(timespec="seconds"),
        "camera_id": o.camera_id,
        "floor_id": o.floor_id,
        "x": round(o.world_x, 2), "y": round(o.world_y, 2),
        "speed": round(o.speed, 2) if o.speed is not None else None,
    } for o in reversed(rows)]


@router.get("/prediction/{global_id}")
def api_prediction(global_id: int, request: Request):
    """Прогноз: ожидаемая позиция + камеры «впереди» с ETA (ТЗ §10)."""
    spatial = _spatial(request)
    if spatial is None:
        raise HTTPException(503, "2.5D-модель выключена (SPATIAL_MODEL_ENABLED)")
    result = spatial.predict(global_id)
    if result is None:
        raise HTTPException(404, "Нет живой траектории для этой личности")
    return result


@router.get("/debug/matches")
def api_debug_matches(request: Request, limit: int = 50):
    """Последние решения матчинга с spatial-оценками и причинами отказов
    (ТЗ §19)."""
    spatial = _spatial(request)
    if spatial is None:
        return {"spatial_model_enabled": False, "decisions": []}
    return {"spatial_model_enabled": True,
            "decisions": spatial.debug_matches(limit)}
