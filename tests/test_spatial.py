"""Тесты 2.5D Spatial World Model: геометрия, траектории, навигация,
scorer, сервис и интеграция с GlobalIdentityManager + REST API.

Запуск (без pytest, только зависимости проекта):

    venv/Scripts/python tests/test_spatial.py      # Windows
    venv/bin/python tests/test_spatial.py          # macOS / Linux

Проверяется:
- гомография pixel↔world (точность, ошибка репроекции, foot point, coverage);
- траектории: скорость/направление по окну, не по одной точке;
- навигационный граф: Дейкстра, время прохода, межэтажные правила;
- spatial-оценки: возможный переход / невозможный (veto) / другой этаж /
  направление / стена;
- сервис: проекция детекций, коммит global_id, троттлинг записи наблюдений;
- интеграция: spatial-veto меняет решение матчинга (новая identity вместо
  ложного слияния), spatial-оценки попадают в payload событий;
- API: этажи, фичи, калибровка, узлы/рёбра, /world.
"""
import asyncio
import json
import math
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from fastapi import HTTPException
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.ai.global_tracker import GlobalIdentityManager
from app.api import spatial as api
from app.database.database import Base
from app.database import models as db_models  # noqa: F401 — регистрация таблиц
from app.database.models import (
    Camera, GlobalEvent, SpatialCameraSetup, SpatialCalibration,
    SpatialFeature, SpatialFloor, SpatialObservation,
)
from app.spatial.geometry import (
    compute_homography, coverage_from_homography, foot_point, pixel_to_world,
)
from app.spatial.matcher import SpatialScorer
from app.spatial.navigation import EdgeInfo, NavigationGraph, NodeInfo
from app.spatial.prediction import predict
from app.spatial.service import SpatialWorldModel
from app.spatial.trajectory import TrajectoryRegistry
from app.spatial.world import WorldHolder, WorldModel

# ------------------------------------------------------------------ стенды

class FakeReID:
    def __init__(self):
        self.vector = None

    def set_person(self, index: int) -> None:
        vec = np.zeros(512, dtype=np.float32)
        vec[index] = 1.0
        self.vector = vec

    def embed(self, crops):
        return np.stack([self.vector] * len(crops))


@dataclass
class Det:
    track_id: int
    employee_id: int = None
    bbox: tuple = (0.3, 0.2, 0.45, 0.85)
    global_id: int = None


class Outcome:
    def __init__(self, *dets):
        self.detections = list(dets)


class FakeRequest:
    """request для прямого вызова API-функций (воркера нет — модель из БД).
    manager — фейковый ридер с кадром (для маркерной калибровки)."""

    def __init__(self, manager=None):
        self.app = SimpleNamespace(
            state=SimpleNamespace(ai_worker=None, manager=manager))


class FakeReader:
    def __init__(self, frame):
        self.buffer = SimpleNamespace(latest=lambda: (1, frame))

    def is_alive(self):
        return True


class FakeManager:
    def __init__(self, reader):
        self._reader = reader

    def get(self, camera_id):
        return self._reader


FRAME = np.zeros((1080, 1920, 3), dtype=np.uint8)
FRAME_SHAPE = (1080, 1920)

_engine = create_engine(
    "sqlite://",
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
)
SessionFactory = sessionmaker(bind=_engine, expire_on_commit=False)

# гомография тестового мира: world = pixel * 0.01 (метры)
H_SCALE = [[0.01, 0.0, 0.0], [0.0, 0.01, 0.0], [0.0, 0.0, 1.0]]


def fresh_db():
    Base.metadata.drop_all(_engine)
    Base.metadata.create_all(_engine)


def seed_world(with_wall=False, with_nav=False):
    """Мир: этаж 1 (z=0), этаж 2 (z=3.5); камеры 1,2 на этаже 1, камера 3
    на этаже 2; все откалиброваны H_SCALE (world = pixel/100)."""
    fresh_db()
    with SessionFactory() as db:
        db.add_all([
            SpatialFloor(id=1, name="1 этаж", z=0.0),
            SpatialFloor(id=2, name="2 этаж", z=3.5),
            Camera(id=1, name="Камера 1", nvr_host="h", username="u",
                   password="p", channel=1),
            Camera(id=2, name="Камера 2", nvr_host="h", username="u",
                   password="p", channel=2),
            Camera(id=3, name="Камера 3", nvr_host="h", username="u",
                   password="p", channel=3),
            SpatialCameraSetup(camera_id=1, floor_id=1, pos_x=1.0, pos_y=1.0),
            SpatialCameraSetup(camera_id=2, floor_id=1, pos_x=20.0, pos_y=1.0),
            SpatialCameraSetup(camera_id=3, floor_id=2, pos_x=1.0, pos_y=1.0),
        ])
        for cam_id in (1, 2, 3):
            db.add(SpatialCalibration(
                camera_id=cam_id, homography=json.dumps(H_SCALE),
                calibration_points="[]", resolution_w=1920, resolution_h=1080,
                reprojection_error=0.0,
            ))
        if with_wall:
            db.add(SpatialFeature(
                floor_id=1, ftype="wall",
                geometry=json.dumps({"start": [0, 7], "end": [30, 7],
                                     "height": 3.2})))
        if with_nav:
            db.add_all([
                SpatialFloor(id=1, name="1 этаж", z=0.0),
            ])
        db.commit()


def load_service(with_wall=False):
    seed_world(with_wall=with_wall)
    return SpatialWorldModel.load(SessionFactory)


def det_at(track_id: int, wx: float, wy: float, employee_id=None) -> Det:
    """Детекция, чей foot point попадает в мировую точку (wx, wy):
    нормализованный bbox подбирается под H_SCALE и кадр 1920×1080."""
    fx = wx * 100 / 1920
    fy = wy * 100 / 1080
    return Det(track_id=track_id, employee_id=employee_id,
               bbox=(fx - 0.05, fy - 0.15, fx + 0.05, fy))


def expect_http(code: int, fn, *args, **kwargs):
    try:
        fn(*args, **kwargs)
    except HTTPException as exc:
        assert exc.status_code == code, \
            f"ожидали HTTP {code}, получили {exc.status_code}"
        return
    raise AssertionError(f"ожидали HTTP {code}, вызов прошёл без ошибки")


# ------------------------------------------------------------------- тесты

TESTS = []


def test(fn):
    TESTS.append(fn)
    return fn


# ------------------------------------------------------------ геометрия

@test
def test_foot_point():
    fp = foot_point((0.3, 0.2, 0.45, 0.85))
    assert fp == (0.375, 0.85), "foot point = низ и центр bbox"


@test
def test_homography_roundtrip():
    points = [
        {"pixel": [421, 712], "world": [4.21, 7.12]},
        {"pixel": [1150, 680], "world": [11.50, 6.80]},
        {"pixel": [1510, 830], "world": [15.10, 8.30]},
        {"pixel": [620, 900], "world": [6.20, 9.00]},
        {"pixel": [900, 500], "world": [9.00, 5.00]},
    ]
    H, error = compute_homography(points)
    assert error < 1e-6, f"ошибка репроекции должна быть ~0, got {error}"
    w = pixel_to_world(H, 421, 712)
    assert abs(w[0] - 4.21) < 1e-6 and abs(w[1] - 7.12) < 1e-6


@test
def test_homography_needs_four_points():
    expect_http_like(ValueError, compute_homography,
                     [{"pixel": [1, 1], "world": [0, 0]}] * 3)


def expect_http_like(exc_type, fn, *args):
    try:
        fn(*args)
    except exc_type:
        return
    raise AssertionError(f"ожидали {exc_type.__name__}")


@test
def test_coverage_from_homography():
    H = np.array(H_SCALE)
    cov = coverage_from_homography(H, 1920, 1080)
    assert cov is not None
    xs = [p[0] for p in cov]
    ys = [p[1] for p in cov]
    assert abs(min(xs)) < 1e-6 and abs(max(xs) - 19.2) < 1e-6
    assert abs(max(ys) - 10.8) < 1e-6


# ------------------------------------------------------------ траектории

@test
def test_trajectory_velocity_window():
    reg = TrajectoryRegistry(maxlen=30, alpha=0.4)
    for i in range(6):                     # движение +X со скоростью 1 м/с
        reg.update(1, 5, 10.0 + i * 0.25, 5.0 + i * 0.25, 5.0)
    snap = reg.state(1, 5)
    assert snap.speed is not None
    assert abs(snap.speed - 1.0) < 0.05, f"скорость ~1 м/с, got {snap.speed}"
    assert abs(snap.direction) < 0.05, "направление +X (angle ~0)"
    # одна точка — скорости нет (оценки не по одной точке)
    reg.update(2, 9, 20.0, 1.0, 1.0)
    assert reg.state(2, 9).speed is None


@test
def test_trajectory_bind_and_last_of():
    reg = TrajectoryRegistry()
    reg.update(1, 5, 10.0, 5.0, 5.0)
    reg.bind(1, 5, 42)
    reg.update(2, 7, 11.0, 8.0, 5.0)
    reg.bind(2, 7, 42)
    last = reg.last_of(42)
    assert last.camera_id == 2 and last.track_id == 7
    reg.prune(100.0, ttl=5.0)
    assert reg.last_of(42) is None, "протухшие траектории чистятся"


# ------------------------------------------------------------ навигация

@test
def test_navigation_graph_paths():
    nodes = [
        NodeInfo(id=1, floor_id=1, ntype="door", name="дверь", x=0, y=0),
        NodeInfo(id=2, floor_id=1, ntype="corridor", name=None, x=3, y=4),
        NodeInfo(id=3, floor_id=2, ntype="room", name=None, x=0, y=0),
        NodeInfo(id=4, floor_id=2, ntype="stairs", name="лестница", x=3, y=4),
    ]
    edges = [
        EdgeInfo(id=1, from_id=1, to_id=2, distance=5.0,
                 min_time=None, max_time=None),
        EdgeInfo(id=2, from_id=4, to_id=3, distance=5.0,
                 min_time=None, max_time=None),
        EdgeInfo(id=3, from_id=2, to_id=4, distance=1.0,
                 min_time=None, max_time=None),
        # межэтажное ребро БЕЗ stairs/elevator — должно игнорироваться
        EdgeInfo(id=4, from_id=1, to_id=3, distance=1.0,
                 min_time=None, max_time=None),
    ]
    nav = NavigationGraph(nodes, edges, max_speed=2.0)
    assert nav.path_length(1, 2) == 5.0
    # 1(этаж1) → 2 → 4(лестница) → 3(этаж2): 5+1+5 = 11 м — единственный
    # законный межэтажный маршрут (прямое ребро 1-3 игнорируется)
    assert nav.path_length(1, 3) == 11.0
    min_t, max_t = nav.travel_window(1, 3)
    assert abs(min_t - 5.5) < 1e-6, "11 м / 2 м/с = 5.5 с"
    assert max_t is not None and max_t >= min_t
    nearest = nav.nearest_node(2.9, 4.1, floor_id=1)
    assert nearest.id == 2


@test
def test_navigation_cross_floor_requires_stairs():
    nodes = [
        NodeInfo(id=1, floor_id=1, ntype="room", name=None, x=0, y=0),
        NodeInfo(id=2, floor_id=2, ntype="room", name=None, x=1, y=1),
    ]
    edges = [EdgeInfo(id=1, from_id=1, to_id=2, distance=2.0,
                      min_time=None, max_time=None)]
    nav = NavigationGraph(nodes, edges)
    assert nav.path_length(1, 2) is None, \
        "межэтажный переход без stairs/elevator запрещён (ТЗ §12)"


# ------------------------------------------------------------- scorer

@test
def test_scorer_possible_transition():
    service = load_service()
    # G#1 виделся на камере 1 в (5,5), появился на камере 2 в (9,5) через 10 с
    service.trajectories.update(1, 5, 100.0, 5.0, 5.0)
    service.trajectories.bind(1, 5, 1)
    service.trajectories.update(2, 7, 110.0, 9.0, 5.0)
    ev = service.scorer.evaluate(1, 2, 7, now=110.0)
    assert ev.available
    assert not ev.veto, "4 м за 10 с — физически возможно"
    assert ev.temporal == 1.0
    assert ev.spatial > 0.0


@test
def test_scorer_impossible_distance_veto():
    service = load_service()
    service.trajectories.update(1, 5, 100.0, 5.0, 5.0)
    service.trajectories.bind(1, 5, 1)
    service.trajectories.update(2, 7, 102.0, 17.0, 5.0)   # 12 м за 2 с
    ev = service.scorer.evaluate(1, 2, 7, now=102.0)
    assert ev.available and ev.veto, "12 м за 2 с — veto"
    assert "impossible" in ev.reason or "too short" in ev.reason


@test
def test_scorer_wrong_floor_veto():
    service = load_service()
    service.trajectories.update(1, 5, 100.0, 5.0, 5.0)
    service.trajectories.bind(1, 5, 1)
    service.trajectories.update(3, 9, 130.0, 5.0, 5.0)    # камера 3 — этаж 2
    ev = service.scorer.evaluate(1, 3, 9, now=130.0)
    assert ev.available and ev.veto and "wrong floor" in ev.reason


@test
def test_scorer_wall_penalty():
    service = load_service(with_wall=True)     # стена y=7 через весь этаж
    service.trajectories.update(1, 5, 100.0, 5.0, 5.0)
    service.trajectories.bind(1, 5, 1)
    service.trajectories.update(2, 7, 110.0, 5.0, 9.0)    # прямая сквозь стену
    ev = service.scorer.evaluate(1, 2, 7, now=110.0)
    assert ev.available and not ev.veto
    assert ev.spatial <= 0.3 * 0.5, \
        "пересечение стены — тяжёлый штраф (но не veto)"


@test
def test_scorer_direction_mismatch():
    service = load_service()
    # G#1 двигался +X; новый трек на камере 2 идёт в обратную сторону
    for i in range(4):
        service.trajectories.update(1, 5, 100.0 + i * 0.3, 5.0 + i * 0.3, 5.0)
    service.trajectories.bind(1, 5, 1)
    for i in range(4):
        service.trajectories.update(2, 7, 110.0 + i * 0.3, 9.0 - i * 0.3, 5.0)
    ev = service.scorer.evaluate(1, 2, 7, now=111.0)
    assert ev.available
    assert ev.direction < 0.5, "движение против направления перехода"


@test
def test_scorer_no_data_is_neutral():
    service = load_service()
    ev = service.scorer.evaluate(999, 2, 7, now=100.0)
    assert not ev.available, "нет траектории — fallback на прежнюю формулу"


# ------------------------------------------------------------- сервис

@test
def test_service_process_and_commit():
    service = load_service()
    det = det_at(5, wx=5.0, wy=5.0)
    outcome = Outcome(det)
    service.process(1, outcome, FRAME_SHAPE, now=100.0)
    snap = service.trajectories.state(1, 5)
    assert snap is not None and snap.position is not None
    x, y = snap.position
    assert abs(x - 5.0) < 0.05 and abs(y - 5.0) < 0.05, \
        "foot point → гомография → мировые координаты"

    det.global_id = 42
    service.commit(1, outcome, now=100.0)
    det2 = det_at(5, wx=5.1, wy=5.0)
    service.process(1, Outcome(det2), FRAME_SHAPE, now=100.2)
    det2.global_id = 42
    service.commit(1, Outcome(det2), now=100.2)
    with SessionFactory() as db:
        rows = db.query(SpatialObservation).filter_by(global_id=42).all()
        assert len(rows) == 1, "троттлинг: 0.2 с < spatial_obs_interval"
        assert rows[0].floor_id == 1
        assert abs(rows[0].world_x - 5.0) < 0.05


@test
def test_service_live_and_prediction():
    service = load_service()
    for i in range(4):
        det = det_at(5, wx=5.0 + i * 0.3, wy=5.0)
        service.process(1, Outcome(det), FRAME_SHAPE, now=100.0 + i * 0.3)
        det.global_id = 7
        service.commit(1, Outcome(det), now=100.0 + i * 0.3)
    people = service.live(now=101.2)
    assert len(people) == 1 and people[0]["global_id"] == 7
    assert people[0]["speed"] is not None and people[0]["speed"] > 0.5
    pred = service.predict(7)
    assert pred is not None and pred["moving"] is True
    assert pred["expected_positions"], "есть точки предсказания"
    # камера 2 видит x>=20 (гомография 0..19.2) — предсказание +X не дойдёт
    # за 30 с при 1 м/с: позиция ~5.9+30=35.9, но зона камеры 2 с x=20
    ahead_ids = [c["camera_id"] for c in pred["cameras_ahead"]]
    assert 2 in ahead_ids, "камера 2 впереди по движению +X"


# ---------------------------------------------------------- интеграция

@test
def test_integration_spatial_veto_prevents_false_merge():
    """Высокий Re-ID score, но физически невозможный переход — создаётся
    НОВАЯ identity, а не ложное слияние (главный критерий ТЗ §22)."""
    service = load_service()
    reid = FakeReID()
    reid.set_person(0)
    manager = GlobalIdentityManager(SessionFactory, reid, topology=None,
                                    spatial=service)
    # кадр 1: камера 1, мир (5,5)
    d1 = det_at(1, wx=5.0, wy=5.0)
    out1 = Outcome(d1)
    service.process(1, out1, FRAME_SHAPE, now=10.0)
    manager.process(1, out1, FRAME, now=10.0)
    service.commit(1, out1, now=10.0)
    assert d1.global_id == 1

    # тот же человек (та же внешность) «появился» на камере 2 через 2 с
    # в 12 метрах — физически невозможно
    d2 = det_at(7, wx=17.0, wy=5.0)
    out2 = Outcome(d2)
    service.process(2, out2, FRAME_SHAPE, now=12.0)
    manager.process(2, out2, FRAME, now=12.0)
    assert d2.global_id == 2, \
        "spatial-veto: невозможный переход → новая identity, не ложный матч"


@test
def test_integration_spatial_scores_in_match():
    """Возможный переход с высокой spatial-оценкой — матч; в payload
    события попадают spatial/temporal/direction оценки (ТЗ §19)."""
    service = load_service()
    reid = FakeReID()
    reid.set_person(0)
    manager = GlobalIdentityManager(SessionFactory, reid, topology=None,
                                    spatial=service)
    # камера 1: два кадра — есть позиция И скорость (+X, 1 м/с)
    for t, wx in ((10.0, 5.0), (10.5, 5.5)):
        d = det_at(1, wx=wx, wy=5.0)
        out = Outcome(d)
        service.process(1, out, FRAME_SHAPE, now=t)
        manager.process(1, out, FRAME, now=t)
        service.commit(1, out, now=t)
    gid = 1

    # камера 2 через 9.5 с в (13,5): ожидаемая позиция (5.2+9.5)≈14.7 — рядом
    d2 = det_at(7, wx=13.0, wy=5.0)
    out2 = Outcome(d2)
    service.process(2, out2, FRAME_SHAPE, now=20.0)
    manager.process(2, out2, FRAME, now=20.0)
    assert d2.global_id == gid, "физически возможный переход — тот же global_id"

    with SessionFactory() as db:
        events = db.query(GlobalEvent).filter_by(
            global_id=gid, event_type="person_seen").all()
        assert events, "событие person_seen записано"
        payload = json.loads(events[-1].payload)
        assert "spatial" in payload and "temporal" in payload \
            and "direction" in payload, \
            "spatial-оценки в payload (отладка матчей ТЗ §19)"
        assert payload["spatial"] >= 0.9, "новая позиция у ожидаемой"
    # решение в debug ring buffer
    decisions = service.debug_matches()
    assert any(d["decision"] == "MATCHED" for d in decisions)


@test
def test_integration_wrong_floor_veto():
    service = load_service()
    reid = FakeReID()
    reid.set_person(1)
    manager = GlobalIdentityManager(SessionFactory, reid, topology=None,
                                    spatial=service)
    d1 = det_at(1, wx=5.0, wy=5.0)
    out1 = Outcome(d1)
    service.process(1, out1, FRAME_SHAPE, now=10.0)
    manager.process(1, out1, FRAME, now=10.0)
    service.commit(1, out1, now=10.0)

    d2 = det_at(9, wx=5.0, wy=5.0)          # камера 3 — ДРУГОЙ этаж
    out2 = Outcome(d2)
    service.process(3, out2, FRAME_SHAPE, now=30.0)
    manager.process(3, out2, FRAME, now=30.0)
    assert d2.global_id != d1.global_id, \
        "разные этажи без stairs/elevator — не может быть тем же человеком"


# ----------------------------------------------------------------- API

@test
def test_api_floors_crud():
    fresh_db()
    request = FakeRequest()
    api.api_create_floor(api.FloorCreate(name="1 этаж", z=0.0),
                         request=request, db=SessionFactory())
    api.api_create_floor(api.FloorCreate(name="2 этаж", z=3.5),
                         request=request, db=SessionFactory())
    floors = api.api_list_floors(db=SessionFactory())
    assert [f["name"] for f in floors] == ["1 этаж", "2 этаж"]

    api.api_update_floor(2, api.FloorUpdate(name="Цоколь", z=-0.5),
                         request=request, db=SessionFactory())
    floors = api.api_list_floors(db=SessionFactory())
    assert floors[1]["name"] == "Цоколь" and floors[1]["z"] == -0.5

    api.api_delete_floor(2, request=request, db=SessionFactory())
    assert len(api.api_list_floors(db=SessionFactory())) == 1
    expect_http(404, api.api_delete_floor, 99, request=request,
                db=SessionFactory())


@test
def test_api_features_validation():
    seed_world()
    request = FakeRequest()
    created = api.api_create_feature(
        1, api.FeatureCreate(
            type="wall", name="стена",
            geometry={"start": [0, 0], "end": [15.5, 0], "height": 3.2}),
        request=request, db=SessionFactory())
    assert created["type"] == "wall"

    updated = api.api_update_feature(
        created["id"], api.FeatureUpdate(geometry={
            "start": [0, 0], "end": [16.0, 0], "height": 3.0}),
        request=request, db=SessionFactory())
    assert updated["geometry"]["end"] == [16.0, 0]

    expect_http(400, api.api_create_feature, 1,
                api.FeatureCreate(type="wall", geometry={"start": [0, 0]}),
                request=request, db=SessionFactory())
    expect_http(400, api.api_create_feature, 1,
                api.FeatureCreate(type="zone", geometry={"points": [[0, 0]]}),
                request=request, db=SessionFactory())
    api.api_delete_feature(created["id"], request=request,
                           db=SessionFactory())
    assert api.api_list_features(1, db=SessionFactory()) == []


@test
def test_api_calibration():
    seed_world()
    request = FakeRequest()
    # 4 точки пола: pixel → world известного преобразования
    points = [
        {"pixel": [421, 712], "world": [4.21, 7.12]},
        {"pixel": [1150, 680], "world": [11.50, 6.80]},
        {"pixel": [1510, 830], "world": [15.10, 8.30]},
        {"pixel": [620, 900], "world": [6.20, 9.00]},
    ]
    payload = api.CalibrationIn(
        points=[api.CalibrationPoint(pixel=p["pixel"], world=p["world"])
                for p in points],
        resolution=[1920, 1080],
    )
    result = api.api_calibrate_camera(1, payload, request=request,
                                      db=SessionFactory())
    assert result["calibrated"] is True
    assert result["reprojection_error"] < 0.01
    assert result["coverage_polygon"], "зона видимости построена из гомографии"

    stored = api.api_get_calibration(1, db=SessionFactory())
    assert stored["calibrated"] is True
    assert len(stored["homography"]) == 3
    assert len(stored["calibration_points"]) == 4

    expect_http(400, api.api_calibrate_camera, 1,
                api.CalibrationIn(points=payload.points[:3],
                                  resolution=[1920, 1080]),
                request=request, db=SessionFactory())

    api.api_delete_calibration(1, request=request, db=SessionFactory())
    assert api.api_get_calibration(1, db=SessionFactory())["calibrated"] is False


@test
def test_api_camera_setup_and_world():
    seed_world()
    request = FakeRequest()
    api.api_update_camera_setup(
        1, api.CameraSetupIn(
            floor_id=1,
            position={"x": 4.2, "y": 8.5, "z": 3.1},
            rotation={"yaw": 180, "pitch": -25, "roll": 0},
            fov={"horizontal": 90, "vertical": 55},
            coverage_polygon=None),
        request=request, db=SessionFactory())
    world = api.api_world(request=request, db=SessionFactory())
    cam = next(c for c in world["cameras"] if c["camera_id"] == 1)
    assert cam["position"]["x"] == 4.2 and cam["floor_id"] == 1
    assert cam["calibrated"] is True, "калибровка из seed_world"
    assert cam["camera_name"] == "Камера 1"
    assert set(world.keys()) >= {"floors", "cameras", "nodes", "edges"}

    # камера без сетапа всё равно видна в /world (можно размещать)
    with SessionFactory() as db:
        db.add(Camera(id=9, name="Новая", nvr_host="h", username="u",
                      password="p", channel=9))
        db.commit()
    world = api.api_world(request=request, db=SessionFactory())
    cam9 = next(c for c in world["cameras"] if c["camera_id"] == 9)
    assert cam9["position"] is None and cam9["calibrated"] is False


@test
def test_api_nodes_edges_and_floor_rules():
    seed_world()
    request = FakeRequest()
    a = api.api_create_node(api.NodeCreate(floor_id=1, type="door",
                                           name="дверь 1", position=[8.4, 12.2]),
                            request=request, db=SessionFactory())
    b = api.api_create_node(api.NodeCreate(floor_id=1, type="corridor",
                                           name=None, position=[10, 15]),
                            request=request, db=SessionFactory())
    c = api.api_create_node(api.NodeCreate(floor_id=2, type="room",
                                           name=None, position=[5, 5]),
                            request=request, db=SessionFactory())
    # расстояние считается автоматически (евклид)
    edge = api.api_create_edge(api.EdgeCreate(from_node=a["id"], to_node=b["id"]),
                               request=request, db=SessionFactory())
    dist = math.hypot(10 - 8.4, 15 - 12.2)
    assert abs(edge["distance"] - dist) < 0.01
    # дубликат запрещён
    expect_http(400, api.api_create_edge,
                api.EdgeCreate(from_node=b["id"], to_node=a["id"]),
                request=request, db=SessionFactory())
    # межэтажное ребро без stairs/elevator — 400
    expect_http(400, api.api_create_edge,
                api.EdgeCreate(from_node=a["id"], to_node=c["id"]),
                request=request, db=SessionFactory())
    # через лестницу — можно
    stairs = api.api_create_node(api.NodeCreate(
        floor_id=1, type="stairs", name="лестница", position=[8, 13]),
        request=request, db=SessionFactory())
    ok_edge = api.api_create_edge(
        api.EdgeCreate(from_node=stairs["id"], to_node=c["id"]),
        request=request, db=SessionFactory())
    assert ok_edge["id"] > 0

    api.api_delete_node(stairs["id"], request=request, db=SessionFactory())
    edges = api.api_list_edges(db=SessionFactory())
    assert all(e["from_node"] != stairs["id"] and e["to_node"] != stairs["id"]
               for e in edges), "рёбра узла удаляются каскадом"


# ----------------------------------------------------- маркеры ArUco

# «истинная» гомография тестового кадра: лёгкая перспектива
H_TRUE = np.array([[0.010, 0.0010, 2.0],
                   [0.0005, 0.0095, 3.0],
                   [1e-6, 5e-7, 1.0]])


def world_of(H, px, py):
    v = H @ np.array([px, py, 1.0])
    return [float(v[0] / v[2]), float(v[1] / v[2])]


def make_marker_frame():
    """Белый кадр 1920×1080 с 6 ArUco-маркерами в известных позициях."""
    from app.spatial.markers import _dictionary
    import cv2
    d = _dictionary()
    frame = np.full((1080, 1920, 3), 255, dtype=np.uint8)
    positions = {0: (300, 300), 1: (900, 250), 2: (1500, 350),
                 3: (400, 800), 4: (1000, 850), 5: (1500, 900)}
    for marker_id, (cx, cy) in positions.items():
        img = cv2.aruco.generateImageMarker(d, marker_id, 200)
        x0, y0 = cx - 100, cy - 100
        frame[y0:y0 + 200, x0:x0 + 200] = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return frame, positions


@test
def test_markers_detection_and_homography():
    """Детект находит все маркеры точно; по ним восстанавливается
    истинная гомография (суть авто-калибровки)."""
    from app.spatial.markers import detect_markers
    frame, positions = make_marker_frame()
    markers = detect_markers(frame)
    assert sorted(m.marker_id for m in markers) == [0, 1, 2, 3, 4, 5]
    for m in markers:
        cx, cy = positions[m.marker_id]
        assert abs(m.pixel[0] - cx) < 2 and abs(m.pixel[1] - cy) < 2, \
            f"центр маркера {m.marker_id}"
    pairs = [{"pixel": list(m.pixel), "world": world_of(H_TRUE, *m.pixel)}
             for m in markers]
    H, error = compute_homography(pairs)
    assert error < 1e-6, "точки согласованы — ошибка ~0"
    # новая точка проецируется как по истинной гомографии
    w = pixel_to_world(H, 960, 540)
    expected = world_of(H_TRUE, 960, 540)
    assert abs(w[0] - expected[0]) < 1e-6 and abs(w[1] - expected[1]) < 1e-6


@test
def test_api_markers_sheet():
    res = api.api_markers_sheet(count=6, marker_cm=10, dpi=150)
    assert res.status_code == 200
    assert res.body[:8] == b"\x89PNG\r\n\x1a\n", "PNG-лист"
    # невместимое — понятная ошибка, а не краш
    expect_http(400, api.api_markers_sheet, count=50, marker_cm=5.0, dpi=150)


@test
def test_marker_pages_pdf():
    """По одному крупному маркеру на страницу: каждая страница детектится,
    PDF структурно валиден, отдаётся как application/pdf."""
    from app.spatial.markers import detect_markers, generate_pages_pdf, \
        render_marker_pages
    pages = render_marker_pages(count=3, marker_cm=18)
    assert len(pages) == 3
    ids = []
    for page in pages:
        found = detect_markers(page)
        assert len(found) == 1, "ровно один маркер на странице"
        ids.append(found[0].marker_id)
    assert ids == [0, 1, 2]
    pdf = generate_pages_pdf(count=3, marker_cm=18)
    assert pdf[:8] == b"%PDF-1.4"
    assert b"%%EOF" in pdf[-32:]
    assert pdf.count(b"/Type /Page ") == 3, "3 страницы"
    # API-вариант one_per_page
    res = api.api_markers_sheet(count=3, marker_cm=18, dpi=150,
                                one_per_page=True)
    assert res.status_code == 200 and res.media_type == "application/pdf"
    assert res.body[:8] == b"%PDF-1.4"


@test
def test_api_auto_calibration_by_markers():
    """Полный цикл: кадр с маркерами + таблица ID→мир → гомография
    сохранена; цепочный замер через эту камеру возвращает координаты."""
    from app.spatial.markers import detect_markers
    seed_world()
    frame, positions = make_marker_frame()
    marker_world = {i: world_of(H_TRUE, *pos) for i, pos in positions.items()}
    request = FakeRequest(FakeManager(FakeReader(frame)))
    payload = api.AutoCalibrationIn(points=[
        api.MarkerWorldPoint(marker_id=i, world=marker_world[i])
        for i in sorted(marker_world)])
    result = api.api_auto_calibrate(1, payload, request=request,
                                    db=SessionFactory())
    assert result["calibrated"] is True
    assert result["reprojection_error"] < 0.01
    assert result["used_markers"] == [0, 1, 2, 3, 4, 5]
    assert result["resolution"] == [1920, 1080]

    stored = api.api_get_calibration(1, db=SessionFactory())
    assert stored["calibrated"] is True

    # цепочка: та же камера «обмеряет» маркеры своей гомографией
    measured = api.api_measure_markers(1, request=request,
                                       db=SessionFactory())
    assert len(measured["markers"]) == 6
    for m in measured["markers"]:
        expected = marker_world[m["marker_id"]]
        assert abs(m["world"][0] - expected[0]) < 0.01
        assert abs(m["world"][1] - expected[1]) < 0.01

    # маркеры не в кадре → понятный 400
    expect_http(400, api.api_auto_calibrate, 1,
                api.AutoCalibrationIn(points=[
                    api.MarkerWorldPoint(marker_id=42, world=[0, 0])] * 4),
                request=request, db=SessionFactory())


# -------------------------------------------------------------------- раннер

def main() -> int:
    print("Тесты 2.5D Spatial World Model")
    failures = 0
    for fn in TESTS:
        try:
            fn()
            print(f"  ok   {fn.__name__}")
        except Exception:
            traceback.print_exc()
            print(f"  FAIL {fn.__name__}")
            failures += 1
    total = len(TESTS)
    print(f"\n{'FAIL' if failures else 'PASS'}: {total - failures}/{total}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
