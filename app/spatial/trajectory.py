"""Мировые траектории локальных треков: EMA-сглаживание, скорость, направление.

Оценки не считаются по одной точке: скорость/направление — конечная разность
по окну ~1 сек (при 5 FPS это 4-5 точек), позиция — EMA.
"""
import math
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Optional

VELOCITY_WINDOW = 1.0      # сек — окно оценки скорости
MIN_SPAN = 0.4             # сек — меньше — скорость не оцениваем


@dataclass
class TrajectorySnapshot:
    """Потокобезопасная копия состояния траектории (для скоринга/API)."""
    camera_id: int
    track_id: int
    global_id: Optional[int]
    position: Optional[tuple[float, float]]   # EMA
    speed: Optional[float]                    # м/с
    direction: Optional[float]                # рад, atan2(vy, vx)
    vx: Optional[float]
    vy: Optional[float]
    time: float                               # monotonic последней точки
    points: int

    def velocity_vector(self) -> Optional[tuple[float, float]]:
        if self.vx is None or self.vy is None:
            return None
        return (self.vx, self.vy)


class TrackTrajectory:
    """История (t, x, y) одного локального трека + EMA и скорость."""

    __slots__ = ("points", "_ema", "_alpha", "global_id", "_velocity")

    def __init__(self, maxlen: int, alpha: float):
        self.points: deque[tuple[float, float, float]] = deque(maxlen=maxlen)
        self._ema: Optional[tuple[float, float]] = None
        self._alpha = alpha
        self.global_id: Optional[int] = None
        self._velocity: Optional[tuple[float, float]] = None

    def add(self, t: float, x: float, y: float) -> None:
        if self._ema is None:
            self._ema = (x, y)
        else:
            a = self._alpha
            self._ema = (a * x + (1 - a) * self._ema[0],
                         a * y + (1 - a) * self._ema[1])
        self.points.append((t, x, y))
        self._velocity = self._estimate_velocity()

    def _estimate_velocity(self) -> Optional[tuple[float, float]]:
        if len(self.points) < 2:
            return None
        last_t = self.points[-1][0]
        first = None
        for p in self.points:                    # первая точка внутри окна
            if last_t - p[0] <= VELOCITY_WINDOW:
                first = p
                break
        if first is None:
            first = self.points[0]
        dt = last_t - first[0]
        if dt < MIN_SPAN:
            return None
        return ((self.points[-1][1] - first[1]) / dt,
                (self.points[-1][2] - first[2]) / dt)

    # -- снимки --

    @property
    def position(self) -> Optional[tuple[float, float]]:
        return self._ema

    @property
    def last_time(self) -> Optional[float]:
        return self.points[-1][0] if self.points else None

    def snapshot(self, camera_id: int, track_id: int) -> TrajectorySnapshot:
        vx, vy = self._velocity if self._velocity else (None, None)
        return TrajectorySnapshot(
            camera_id=camera_id, track_id=track_id, global_id=self.global_id,
            position=self._ema,
            speed=math.hypot(vx, vy) if vx is not None else None,
            direction=math.atan2(vy, vx) if vx is not None else None,
            vx=vx, vy=vy,
            time=self.points[-1][0] if self.points else 0.0,
            points=len(self.points),
        )


class TrajectoryRegistry:
    """Все траектории: по ключу (camera, track) + индекс последнего трека
    каждой глобальной личности. Потокобезопасно (RLock)."""

    def __init__(self, maxlen: int = 30, alpha: float = 0.4):
        self._tracks: dict[tuple[int, int], TrackTrajectory] = {}
        self._by_global: dict[int, tuple[int, int]] = {}
        self._lock = threading.RLock()
        self._maxlen = maxlen
        self._alpha = alpha

    # ------------------------------------------------------- запись

    def update(self, camera_id: int, track_id: int,
               t: float, x: float, y: float) -> None:
        with self._lock:
            key = (camera_id, track_id)
            traj = self._tracks.get(key)
            if traj is None:
                traj = TrackTrajectory(self._maxlen, self._alpha)
                self._tracks[key] = traj
            traj.add(t, x, y)

    def bind(self, camera_id: int, track_id: int, global_id: int) -> None:
        """Трек привязан к global_id (после матча) — становится «последним
        наблюдением» этой личности."""
        with self._lock:
            traj = self._tracks.get((camera_id, track_id))
            if traj is not None:
                traj.global_id = global_id
            self._by_global[global_id] = (camera_id, track_id)

    def prune(self, now: float, ttl: float) -> None:
        """Убрать траектории без обновлений дольше ttl (трек потерян)."""
        with self._lock:
            stale = [key for key, traj in self._tracks.items()
                     if not traj.points or now - traj.points[-1][0] > ttl]
            for key in stale:
                traj = self._tracks.pop(key)
                if traj.global_id is not None:
                    if self._by_global.get(traj.global_id) == key:
                        del self._by_global[traj.global_id]

    # -------------------------------------------------------- чтение

    def state(self, camera_id: int, track_id: int) -> Optional[TrajectorySnapshot]:
        with self._lock:
            traj = self._tracks.get((camera_id, track_id))
            return traj.snapshot(camera_id, track_id) if traj else None

    def last_of(self, global_id: int) -> Optional[TrajectorySnapshot]:
        """Последний трек глобальной личности (для скоринга матчинга)."""
        with self._lock:
            key = self._by_global.get(global_id)
            if key is None:
                return None
            traj = self._tracks.get(key)
            return traj.snapshot(*key) if traj else None

    def history(self, global_id: int) -> list[dict]:
        """Точки последнего трека личности (короткая живая история)."""
        with self._lock:
            key = self._by_global.get(global_id)
            traj = self._tracks.get(key) if key else None
            if traj is None:
                return []
            return [{"t": t, "x": x, "y": y} for t, x, y in traj.points]

    def live(self, max_age: float = 5.0,
             now: Optional[float] = None) -> list[TrajectorySnapshot]:
        """Активные траектории с global_id (для карты/API). now — monotonic
        (None = текущий момент; тесты передают своё время)."""
        now = time.monotonic() if now is None else now
        out = []
        with self._lock:
            for key, traj in self._tracks.items():
                if traj.global_id is None or not traj.points:
                    continue
                if now - traj.points[-1][0] > max_age:
                    continue
                out.append(traj.snapshot(*key))
        return out
