"""Предсказание перемещения человека (ТЗ §10): позиция + скорость +
навигационный граф → ожидаемая позиция и камеры «впереди» с ETA.

Не ищем человека среди всех камер: система подсказывает, в какую зону
видимости и когда он может попасть — это дополнительный сигнал для
global identity matching и для карты (пунктир предсказания).
"""
import math
from typing import Optional

from app.config import settings
from app.spatial.geometry import point_in_polygon, point_polygon_distance
from app.spatial.trajectory import TrajectorySnapshot
from app.spatial.world import WorldModel

# скорость ниже которой считается, что человек стоит
MIN_MOVING_SPEED = 0.3
# горизонт предсказания, сек
HORIZON = 30.0
# шаги предсказания для пунктира на карте
PREDICT_STEPS = (2.0, 5.0, 10.0, HORIZON)
# сколько камер «впереди» отдавать
MAX_CAMERAS_AHEAD = 5
# чтобы камера считалась «впереди», направление на неё не должно быть
# противоположно направлению движения (косинус больше порога)
AHEAD_COS_THRESHOLD = -0.2


def predict(world: WorldModel, state: TrajectorySnapshot,
            horizon: float = HORIZON) -> Optional[dict]:
    """Прогноз по последнему состоянию траектории личности.

    Возвращает позицию, скорость, точки предсказания и камеры с ETA либо
    None, если у личности нет мировой позиции.
    """
    if state is None or state.position is None:
        return None
    x, y = state.position
    floor_id = world.floor_of_camera(state.camera_id)
    speed = state.speed or 0.0
    v = state.velocity_vector()
    moving = v is not None and speed >= MIN_MOVING_SPEED

    result = {
        "camera_id": state.camera_id,
        "floor_id": floor_id,
        "position": {"x": round(x, 2), "y": round(y, 2)},
        "speed": round(speed, 2),
        "direction": round(state.direction, 3) if state.direction is not None
                     else None,
        "moving": moving,
        "expected_positions": [],
        "cameras_ahead": [],
    }

    if not moving:
        # человек стоит: камеры, которые его сейчас видят или рядом
        result["cameras_ahead"] = _cameras_nearby(world, x, y, floor_id)
        return result

    vx, vy = v
    # точки предсказания — линейная экстраполяция по скорости
    for t in PREDICT_STEPS:
        tt = min(t, horizon)
        result["expected_positions"].append({
            "t": round(tt, 1),
            "x": round(x + vx * tt, 2),
            "y": round(y + vy * tt, 2),
        })

    end_x = x + vx * horizon
    end_y = y + vy * horizon
    result["cameras_ahead"] = _cameras_ahead(
        world, x, y, end_x, end_y, floor_id, speed)
    return result


def _cameras_nearby(world: WorldModel, x: float, y: float,
                    floor_id: Optional[int]) -> list[dict]:
    """Камеры, чья зона видимости содержит точку (или ближайшие по графу)."""
    out = []
    for camera_id in list(world.setups.keys()):
        cov = world.coverage(camera_id)
        cam_floor = world.floor_of_camera(camera_id)
        if cov and (floor_id is None or cam_floor == floor_id):
            dist = point_polygon_distance(x, y, cov)
            out.append({"camera_id": camera_id, "eta": None,
                        "distance": round(dist, 2)})
    out.sort(key=lambda c: c["distance"])
    return out[:MAX_CAMERAS_AHEAD]


def _cameras_ahead(world: WorldModel, x: float, y: float,
                   end_x: float, end_y: float, floor_id: Optional[int],
                   speed: float) -> list[dict]:
    """Камеры в направлении движения с ETA = расстояние до их зоны видимости
    / скорость. Направление на камеру не должно быть против движения."""
    dx, dy = end_x - x, end_y - y
    path_len = math.hypot(dx, dy)
    if path_len < 1e-6:
        return []
    out = []
    for camera_id in list(world.setups.keys()):
        cov = world.coverage(camera_id)
        if not cov:
            continue
        cam_floor = world.floor_of_camera(camera_id)
        if floor_id is not None and cam_floor != floor_id:
            continue
        # расстояние до зоны видимости — по линии движения
        dist = _distance_along_segment_to_polygon(x, y, end_x, end_y, cov)
        if dist is None:
            continue
        # «впереди»: центр зоны не позади точки старта
        cx = sum(p[0] for p in cov) / len(cov)
        cy = sum(p[1] for p in cov) / len(cov)
        to_cam = (cx - x, cy - y)
        cos = ((dx * to_cam[0] + dy * to_cam[1])
               / (path_len * max(1e-6, math.hypot(*to_cam))))
        if cos < AHEAD_COS_THRESHOLD:
            continue
        eta = dist / max(MIN_MOVING_SPEED, speed)
        out.append({"camera_id": camera_id, "eta": round(eta, 1),
                    "distance": round(dist, 2)})
    out.sort(key=lambda c: c["eta"])
    return out[:MAX_CAMERAS_AHEAD]


def _distance_along_segment_to_polygon(x1, y1, x2, y2, polygon
                                       ) -> Optional[float]:
    """Расстояние от начала отрезка до первого пересечения с полигоном:
    0 — старт внутри полигона; None — отрезок полигон не пересекает."""
    if point_in_polygon(x1, y1, polygon):
        return 0.0
    if point_in_polygon(x2, y2, polygon):
        dist = math.hypot(x2 - x1, y2 - y1)
        return dist
    # двоичный поиск пересечения: сегмент прямой, полигоны выпуклые/вогнутые —
    # проверяем середины, достаточно приближения для ETA
    lo, hi = 0.0, 1.0
    if not _segment_hits_polygon(x1, y1, x2, y2, polygon):
        return None
    for _ in range(12):
        mid = (lo + hi) / 2.0
        mx, my = x1 + (x2 - x1) * mid, y1 + (y2 - y1) * mid
        if point_in_polygon(mx, my, polygon):
            hi = mid
        else:
            lo = mid
    t = (lo + hi) / 2.0
    return math.hypot(x2 - x1, y2 - y1) * t


def _segment_hits_polygon(x1, y1, x2, y2, polygon) -> bool:
    """Пересекает ли отрезок хотя бы одну сторону полигона."""
    from app.spatial.geometry import segments_intersect
    n = len(polygon)
    for i in range(n):
        if segments_intersect((x1, y1), (x2, y2),
                              polygon[i], polygon[(i + 1) % n]):
            return True
    return False
