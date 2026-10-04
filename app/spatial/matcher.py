"""Spatial-оценки для межкамерного матчинга: «мог ли человек физически
оказаться здесь?» (ТЗ §9-§12).

Три оценки (0..1) + жёсткие ограничения (veto):

  temporal  — хватило ли времени: расстояние (прямое или по навигационному
              графу) / максимальная скорость человека = минимальное время;
              быстрее физически невозможно → veto;
  spatial   — близость новой позиции к ожидаемой (позиция + скорость·dt),
              штраф за «прямую сквозь стену»;
  direction — совпадает ли направление движения с направлением перехода.

Разные этажи допускаются только при наличии маршрута через stairs/elevator
(иначе veto «wrong floor»). Нет spatial-данных (камера не откалибрована,
первый трек) — available=False: матчинг падает на прежнюю формулу без
новых слагаемых и без ложных запретов.
"""
import math
from dataclasses import dataclass, field
from typing import Optional

from app.config import settings
from app.spatial.geometry import segments_intersect
from app.spatial.trajectory import TrajectoryRegistry, TrajectorySnapshot
from app.spatial.world import WorldHolder

# допуск к минимальному времени (шум гомографии ~0.2-0.5 м на позицию)
MIN_TIME_TOLERANCE = 0.8
# дальше этого EMA-экстраполяция позиции не распространяется (человек мог
# остановиться по пути)
MAX_EXTRAPOLATION = 15.0
# окно «время ещё разумное»: temporal=1.0, дальше линейное затухание к ttl
REASONABLE_TIME = 60.0
# минимальное перемещение, при котором направление перехода имеет смысл
MIN_MOVE_FOR_DIRECTION = 0.5
# штраф за пересечение стены прямой линией (стены могут быть неполными —
# поэтому штраф, а не veto)
WALL_PENALTY = 0.3


@dataclass
class MatchEvaluation:
    available: bool                                  # False → прежняя формула
    spatial: float = 0.0
    temporal: float = 0.0
    direction: float = 0.0
    veto: bool = False
    reason: str = ""
    details: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "spatial": round(self.spatial, 3),
            "temporal": round(self.temporal, 3),
            "direction": round(self.direction, 3),
            "veto": self.veto,
            "reason": self.reason,
        }


class SpatialScorer:
    """Вызывается GlobalIdentityManager на каждого кандидата при связывании
    нового трека. Только чтение состояния — ничего не решает сам."""

    def __init__(self, holder: WorldHolder, trajectories: TrajectoryRegistry):
        self._holder = holder
        self._trajectories = trajectories

    def evaluate(self, identity_gid: int, new_camera: int,
                 new_track: int, now: float) -> MatchEvaluation:
        world = self._holder.current
        last = self._trajectories.last_of(identity_gid)
        new = self._trajectories.state(new_camera, new_track)
        if (last is None or last.position is None
                or new is None or new.position is None):
            return MatchEvaluation(available=False)

        dt = max(0.0, now - last.time)
        lx, ly = last.position
        nx, ny = new.position
        straight = math.hypot(nx - lx, ny - ly)
        floor_last = world.floor_of_camera(last.camera_id)
        floor_new = world.floor_of_camera(new_camera)
        same_floor = (floor_last is None or floor_new is None
                      or floor_last == floor_new)

        # --- расстояние: прямое (нижняя граница) или маршрут по графу ---
        travel = straight
        graph_path = None
        if not world.nav.empty:
            node_last = world.nav.nearest_node(lx, ly, floor_last)
            node_new = world.nav.nearest_node(nx, ny, floor_new)
            if node_last is not None and node_new is not None:
                graph_path = world.nav.path_length(node_last.id, node_new.id)
                if graph_path is not None:
                    travel = max(straight, graph_path)

        # --- межэтажный переход только через stairs/elevator (§12) ---
        if floor_last is not None and floor_new is not None \
                and floor_last != floor_new:
            if graph_path is None:
                return MatchEvaluation(
                    available=True, veto=True,
                    reason=f"wrong floor: этаж {floor_last} → {floor_new} "
                           f"без stairs/elevator",
                    details=self._details(last, new, dt, straight, None),
                )

        # --- temporal: хватило ли времени (§11) ---
        min_time = travel / max(0.5, settings.spatial_max_speed)
        if dt < MIN_TIME_TOLERANCE * min_time:
            return MatchEvaluation(
                available=True, veto=True,
                reason=(f"distance impossible: {travel:.1f} м минимум за "
                        f"{min_time:.1f} с, прошло {dt:.1f} с"),
                details=self._details(last, new, dt, straight, min_time),
            )
        temporal = self._temporal_score(dt, min_time)

        # --- spatial: близость к ожидаемой позиции (§10) ---
        ex, ey = self._expected_position(last, dt)
        dist_to_expected = math.hypot(nx - ex, ny - ey)
        tol = max(1.5, 0.5 * (last.speed or 0.0) * dt)
        spatial = 1.0 if dist_to_expected <= tol else max(
            0.0, 1.0 - (dist_to_expected - tol) / max(tol, 5.0))
        # прямая «сквозь стену» без графа — тяжёлый штраф
        wall_note = ""
        if graph_path is None and same_floor and floor_last is not None:
            if self._crosses_wall(world, floor_last, (lx, ly), (nx, ny)):
                spatial *= WALL_PENALTY
                wall_note = " (прямая пересекает стену)"

        # --- direction: направление движения vs направление перехода ---
        direction = self._direction_score(last, new)

        return MatchEvaluation(
            available=True, spatial=spatial, temporal=temporal,
            direction=direction,
            reason=wall_note.strip(" ()"),
            details=self._details(last, new, dt, travel, min_time,
                                  expected=(ex, ey)),
        )

    # ------------------------------------------------------------ детали

    @staticmethod
    def _temporal_score(dt: float, min_time: float) -> float:
        """1.0 в разумном окне, линейное затухание к gallery_ttl; в зоне
        допуска (0.8-1.0 от min_time) — пропорционально."""
        if dt < min_time:
            return max(0.0, dt / min_time)
        reasonable = max(min_time * 10.0, REASONABLE_TIME)
        ttl = settings.global_gallery_ttl
        if dt <= reasonable:
            return 1.0
        if ttl <= reasonable:
            return 0.0
        return max(0.0, 1.0 - (dt - reasonable) / (ttl - reasonable))

    @staticmethod
    def _expected_position(last: TrajectorySnapshot, dt: float
                           ) -> tuple[float, float]:
        """Куда человек должен был дойти: позиция + скорость·dt (экстраполяция
        ограничена — по пути он мог остановиться)."""
        lx, ly = last.position
        v = last.velocity_vector()
        if v is None or dt <= 0:
            return (lx, ly)
        dist = math.hypot(v[0], v[1]) * dt
        if dist <= MAX_EXTRAPOLATION:
            return (lx + v[0] * dt, ly + v[1] * dt)
        scale = MAX_EXTRAPOLATION / dist
        return (lx + v[0] * dt * scale, ly + v[1] * dt * scale)

    @staticmethod
    def _direction_score(last: TrajectorySnapshot,
                         new: TrajectorySnapshot) -> float:
        """Косинус между скоростью нового трека и направлением перехода;
        нет данных — нейтрально 0.5."""
        v = new.velocity_vector()
        if v is None or last.position is None or new.position is None:
            return 0.5
        mx = new.position[0] - last.position[0]
        my = new.position[1] - last.position[1]
        if math.hypot(mx, my) < MIN_MOVE_FOR_DIRECTION:
            return 0.5
        vlen = math.hypot(v[0], v[1])
        if vlen < 1e-6:
            return 0.5
        cos = (v[0] * mx + v[1] * my) / (vlen * math.hypot(mx, my))
        return max(0.0, min(1.0, (cos + 1.0) / 2.0))

    @staticmethod
    def _crosses_wall(world, floor_id: int,
                      a: tuple[float, float], b: tuple[float, float]) -> bool:
        for w1, w2 in world.walls(floor_id):
            if segments_intersect(a, b, w1, w2):
                return True
        return False

    @staticmethod
    def _details(last: TrajectorySnapshot, new: TrajectorySnapshot,
                 dt: float, travel: Optional[float], min_time: Optional[float],
                 expected=None) -> dict:
        out = {
            "dt": round(dt, 2),
            "last_position": [round(last.position[0], 2),
                              round(last.position[1], 2)],
            "new_position": [round(new.position[0], 2),
                             round(new.position[1], 2)],
            "last_camera": last.camera_id,
            "new_camera": new.camera_id,
        }
        if travel is not None:
            out["distance"] = round(travel, 2)
        if min_time is not None:
            out["min_time"] = round(min_time, 2)
        if expected is not None:
            out["expected_position"] = [round(expected[0], 2),
                                        round(expected[1], 2)]
        return out
