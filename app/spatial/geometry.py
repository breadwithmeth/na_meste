"""Геометрия 2.5D-модели: foot point, гомография pixel↔world, полигоны.

Основная система координат — метрические X/Y на полу этажа. Z (высота)
хранится в модели (этажи, камеры, стены), но трекинг людей работает в X/Y.
"""
import math
from typing import Optional

import cv2
import numpy as np

# Гомография «срывается» на бесконечность у линии горизонта: точки с |w|
# меньше EPS или дальше MAX_WORLD_DIST метров считаются непроецируемыми.
_EPS = 1e-6
MAX_WORLD_DIST = 10_000.0


def foot_point(bbox) -> tuple[float, float]:
    """Нижняя центральная точка bbox — приблизительно положение ног человека
    на полу. НЕ центр bbox: (x1+x2)/2 по горизонтали и y2 (низ) по вертикали.

    Работает и с нормализованными (0..1), и с пиксельными координатами.
    """
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) / 2.0, float(y2))


def compute_homography(points: list[dict]) -> tuple[np.ndarray, float]:
    """Гомография pixel → world по ≥4 парам точек на полу.

    points: [{"pixel": [px, py], "world": [wx, wy]}, ...]
    Возвращает (H 3×3, средняя ошибка репроекции в метрах). При ≥6 точках
    используется RANSAC (защита от промаха при клике), иначе МНК.
    """
    if len(points) < 4:
        raise ValueError("Нужно минимум 4 пары точек для гомографии")
    src = np.array([[p["pixel"][0], p["pixel"][1]] for p in points],
                   dtype=np.float64)
    dst = np.array([[p["world"][0], p["world"][1]] for p in points],
                   dtype=np.float64)
    H = None
    if len(points) >= 6:
        # порог в метрах (dst) — точка дальше 0.5 м от модели считается промахом
        H, _mask = cv2.findHomography(src, dst, cv2.RANSAC, 0.5)
    if H is None:
        H, _mask = cv2.findHomography(src, dst, 0)
    if H is None:
        raise ValueError("Не удалось вычислить гомографию (точки вырождены?)")
    return H, reprojection_error(H, src, dst)


def reprojection_error(H: np.ndarray, src: np.ndarray, dst: np.ndarray) -> float:
    """Средняя евклидова ошибка проекции pixel→world по заданным парам (м)."""
    total, count = 0.0, 0
    for (px, py), (wx, wy) in zip(src, dst):
        proj = pixel_to_world(H, px, py)
        if proj is None:
            continue
        total += math.hypot(proj[0] - wx, proj[1] - wy)
        count += 1
    return total / count if count else float("inf")


def pixel_to_world(H: np.ndarray, px: float, py: float
                   ) -> Optional[tuple[float, float]]:
    """Спроецировать пиксель кадра на пол. None — точка на бесконечности
    (камера смотрит вдоль плоскости пола)."""
    v = H @ np.array([px, py, 1.0])
    w = float(v[2])
    if abs(w) < _EPS:
        return None
    x, y = float(v[0] / w), float(v[1] / w)
    if math.hypot(x, y) > MAX_WORLD_DIST:
        return None
    return (x, y)


def world_to_pixel(H: np.ndarray, wx: float, wy: float
                   ) -> Optional[tuple[float, float]]:
    """Обратная проекция мировой точки в пиксель кадра (для отладки)."""
    try:
        H_inv = np.linalg.inv(H)
    except np.linalg.LinAlgError:
        return None
    v = H_inv @ np.array([wx, wy, 1.0])
    w = float(v[2])
    if abs(w) < _EPS:
        return None
    px, py = float(v[0] / w), float(v[1] / w)
    if math.hypot(px, py) > 1e7:
        return None
    return (px, py)


def coverage_from_homography(H: np.ndarray, width: int, height: int
                             ) -> Optional[list[tuple[float, float]]]:
    """Полигон видимой камером области пола: проекция углов кадра + выпуклая
    оболочка. None — если хотя бы один угол уходит в бесконечность или «за
    спину» проекции (горизонт внутри кадра): тогда зона задаётся вручную.
    """
    corners = [(0, 0), (width, 0), (width, height), (0, height)]
    pts = []
    for cx, cy in corners:
        v = H @ np.array([cx, cy, 1.0])
        w = float(v[2])
        if abs(w) < 1e-3 or w < 0:
            return None
        x, y = float(v[0] / w), float(v[1] / w)
        if math.hypot(x, y) > MAX_WORLD_DIST:
            return None
        pts.append((x, y))
    hull = cv2.convexHull(np.array(pts, dtype=np.float32))
    return [(float(p[0][0]), float(p[0][1])) for p in hull]


# ------------------------------------------------------------------ полигоны

def point_in_polygon(x: float, y: float, polygon) -> bool:
    """Лучевая проба: точка внутри полигона [[x,y], ...]."""
    n = len(polygon)
    if n < 3:
        return False
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = polygon[i][0], polygon[i][1]
        xj, yj = polygon[j][0], polygon[j][1]
        if (yi > y) != (yj > y):
            x_cross = (xj - xi) * (y - yi) / (yj - yi) + xi
            if x < x_cross:
                inside = not inside
        j = i
    return inside


def point_polygon_distance(x: float, y: float, polygon) -> float:
    """Расстояние от точки до полигона: 0 если внутри, иначе до ближайшей
    стороны (для ETA до зоны видимости камеры)."""
    poly = [(float(p[0]), float(p[1])) for p in polygon]
    if point_in_polygon(x, y, poly):
        return 0.0
    best = float("inf")
    n = len(poly)
    for i in range(n):
        best = min(best, _point_segment_distance(x, y, poly[i], poly[(i + 1) % n]))
    return best


def _point_segment_distance(px: float, py: float, a, b) -> float:
    ax, ay = a
    bx, by = b
    dx, dy = bx - ax, by - ay
    length2 = dx * dx + dy * dy
    if length2 < 1e-12:
        return math.hypot(px - ax, py - ay)
    t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / length2))
    return math.hypot(px - (ax + t * dx), py - (ay + t * dy))


def segments_intersect(p1, p2, p3, p4) -> bool:
    """Пересекаются ли отрезки p1p2 и p3p4 (для проверки «прямая сквозь
    стену» в spatial-оценке)."""
    def orient(a, b, c):
        return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
    d1 = orient(p3, p4, p1)
    d2 = orient(p3, p4, p2)
    d3 = orient(p1, p2, p3)
    d4 = orient(p1, p2, p4)
    if ((d1 > 0) != (d2 > 0)) and ((d3 > 0) != (d4 > 0)):
        return True
    return False


def polygon_bounds(polygon) -> tuple[float, float, float, float]:
    """(min_x, min_y, max_x, max_y) полигона/набора точек."""
    xs = [float(p[0]) for p in polygon]
    ys = [float(p[1]) for p in polygon]
    return min(xs), min(ys), max(xs), max(ys)
