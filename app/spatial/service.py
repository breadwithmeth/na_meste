"""SpatialWorldModel — рантайм-сервис 2.5D-модели в AI-воркере.

Вызывается из цикла воркера в два прохода:

    process()  ДО GlobalIdentityManager.process — спроецировать foot points
               детекций на пол, чтобы у матчинга уже были мировые координаты
               нового трека в его же первом кадре;
    commit()   ПОСЛЕ — проставить global_id в реестр траекторий и записать
               SpatialObservation в БД с троттлингом.

Ошибки глотаются с логом: spatial-слой не должен ронять pipeline (тот же
принцип, что у GlobalIdentityManager). Пишет в БД из AI-потока — SQLite WAL
это допускает (как _persist_observation в global_tracker).
"""
import logging
import threading
from collections import deque
from datetime import timedelta
from typing import Optional

from app.config import settings
from app.spatial.geometry import foot_point, pixel_to_world
from app.spatial.matcher import MatchEvaluation, SpatialScorer
from app.spatial.prediction import predict
from app.spatial.trajectory import TrajectoryRegistry, TrajectorySnapshot
from app.spatial.world import WorldHolder, WorldModel
from app.services.presence_service import utcnow

logger = logging.getLogger("app.spatial.service")

PRUNE_INTERVAL = 10.0          # сек между чистками протухших траекторий
LIVE_MAX_AGE = 5.0             # сек — траектория считается «живой»


class SpatialWorldModel:
    """Потокобезопасный фасад над миром + траекториями + скорером."""

    def __init__(self, session_factory, holder: WorldHolder,
                 trajectories: TrajectoryRegistry):
        self._session_factory = session_factory
        self._holder = holder
        self.trajectories = trajectories
        self.scorer = SpatialScorer(holder, trajectories)
        self._lock = threading.RLock()
        self._last_obs_write: dict[tuple[int, int], float] = {}
        self._last_prune = 0.0
        self._debug_matches: deque[dict] = deque(maxlen=100)

    # ----------------------------------------------------------- фабрика

    @classmethod
    def load(cls, session_factory) -> "SpatialWorldModel":
        """Загрузить мир из БД. Модель может быть пустой (нет этажей/камер) —
        все оценки тогда нейтральны, matcher вернёт available=False."""
        with session_factory() as db:
            world = WorldModel.from_session(db,
                                            settings.spatial_max_speed)
        holder = WorldHolder(world)
        registry = TrajectoryRegistry(maxlen=settings.spatial_trajectory_len,
                                      alpha=settings.spatial_ema_alpha)
        instance = cls(session_factory, holder, registry)
        logger.info("Spatial model: этажей=%d, камер с сетапом=%d, "
                    "откалибровано=%d, узлов навигации=%d",
                    len(world.floors), len(world.setups),
                    len(world.calibrations), 0 if world.nav.empty
                    else len(world.nav.nodes()))
        return instance

    @property
    def world(self) -> WorldModel:
        return self._holder.current

    # ------------------------------------------------ цикл AI-воркера

    def process(self, camera_id: int, outcome, frame_shape, now: float) -> None:
        """Спроецировать детекции камеры на пол (до матчинга).

        frame_shape — (высота, ширина) кадра; пиксели пересчитываются в
        систему координат калибровки (устойчиво к смене разрешения потока).
        """
        try:
            world = self.world
            calib = world.calibration(camera_id)
            if calib is None:
                self._maybe_prune(now)
                return  # камера не откалибрована — мировых координат нет
            frame_h, frame_w = frame_shape
            if frame_w <= 0 or frame_h <= 0:
                return
            sx = calib.resolution[0] / frame_w
            sy = calib.resolution[1] / frame_h
            for det in outcome.detections:
                fx, fy = foot_point(det.bbox)         # нормализованные 0..1
                world_pt = pixel_to_world(calib.homography, fx * frame_w * sx,
                                          fy * frame_h * sy)
                if world_pt is None:
                    continue
                self.trajectories.update(camera_id, det.track_id, now,
                                         world_pt[0], world_pt[1])
            self._maybe_prune(now)
        except Exception:
            logger.exception("Camera %d: ошибка spatial-проекции — "
                             "трекинг не затронут", camera_id)

    def commit(self, camera_id: int, outcome, now: float) -> None:
        """После матчинга: привязать global_id и записать наблюдения."""
        try:
            world = self.world
            floor_id = world.floor_of_camera(camera_id)
            for det in outcome.detections:
                if det.global_id is None:
                    continue
                self.trajectories.bind(camera_id, det.track_id, det.global_id)
                state = self.trajectories.state(camera_id, det.track_id)
                if state is None or state.position is None:
                    continue
                key = (camera_id, det.track_id)
                if now - self._last_obs_write.get(key, 0.0) \
                        < settings.spatial_obs_interval:
                    continue
                self._last_obs_write[key] = now
                self._persist_observation(det.global_id, state, floor_id)
        except Exception:
            logger.exception("Camera %d: ошибка spatial-commit", camera_id)

    # ------------------------------------------------------ для матчинга

    def evaluate(self, identity_gid: int, new_camera: int, new_track: int,
                 now: float) -> Optional[MatchEvaluation]:
        """Spatial-оценки кандидата (вызывает GlobalIdentityManager)."""
        try:
            return self.scorer.evaluate(identity_gid, new_camera, new_track,
                                        now)
        except Exception:
            logger.exception("Spatial: ошибка evaluate(gid=%d)", identity_gid)
            return None

    def record_match_decision(self, decision: dict) -> None:
        """Решение матчинга в ring buffer для /api/spatial/debug/matches."""
        with self._lock:
            self._debug_matches.append({
                "wall_time": utcnow().isoformat(timespec="seconds"),
                **decision,
            })

    def debug_matches(self, limit: int = 50) -> list[dict]:
        with self._lock:
            return list(self._debug_matches)[-limit:][::-1]

    # ---------------------------------------------------------- API/UI

    def live(self, now: float = None) -> list[dict]:
        """Текущие люди в мировых координатах (для карты)."""
        world = self.world
        out = []
        for snap in self.trajectories.live(max_age=LIVE_MAX_AGE, now=now):
            out.append({
                "global_id": snap.global_id,
                "camera_id": snap.camera_id,
                "track_id": snap.track_id,
                "floor_id": world.floor_of_camera(snap.camera_id),
                "x": round(snap.position[0], 2) if snap.position else None,
                "y": round(snap.position[1], 2) if snap.position else None,
                "speed": round(snap.speed, 2) if snap.speed is not None
                         else None,
                "direction": round(snap.direction, 2)
                             if snap.direction is not None else None,
            })
        return out

    def predict(self, global_id: int) -> Optional[dict]:
        """Прогноз перемещения личности (позиция + скорость + граф)."""
        state = self.trajectories.last_of(global_id)
        if state is None:
            return None
        return predict(self.world, state)

    def person_history(self, global_id: int, limit: int = 500) -> list[dict]:
        """История позиций личности: живая память + БД (старое впереди)."""
        from app.database.models import SpatialObservation
        rows = []
        try:
            with self._session_factory() as db:
                query = db.query(SpatialObservation).filter(
                    SpatialObservation.global_id == global_id,
                ).order_by(SpatialObservation.created_at.desc()).limit(limit)
                for o in query.all():
                    rows.append({
                        "time": o.created_at.isoformat(timespec="seconds"),
                        "camera_id": o.camera_id,
                        "floor_id": o.floor_id,
                        "x": round(o.world_x, 2),
                        "y": round(o.world_y, 2),
                        "speed": round(o.speed, 2) if o.speed is not None
                                 else None,
                    })
        except Exception:
            logger.exception("Spatial: не удалось прочитать историю gid=%d",
                             global_id)
        rows.reverse()
        # живой хвост последнего трека добавляем, если он новее записей БД
        live_pts = self.trajectories.history(global_id)
        if live_pts:
            rows.extend({"time": None, "camera_id": None, "floor_id": None,
                         "x": round(p["x"], 2), "y": round(p["y"], 2),
                         "speed": None} for p in live_pts)
        return rows

    # ------------------------------------------------------- обслуживание

    def reload_world(self) -> None:
        """Перечитать мир из БД (после CRUD через API)."""
        with self._session_factory() as db:
            world = WorldModel.from_session(db, settings.spatial_max_speed)
        self._holder.replace(world)
        logger.info("Spatial model перезагружен: этажей=%d, камер=%d, "
                    "откалибровано=%d", len(world.floors), len(world.setups),
                    len(world.calibrations))

    def cleanup(self) -> None:
        """Удалить spatial_observations старше spatial_keep_days."""
        from app.database.models import SpatialObservation
        cutoff = utcnow() - timedelta(days=settings.spatial_keep_days)
        try:
            with self._session_factory() as db:
                db.query(SpatialObservation).filter(
                    SpatialObservation.created_at < cutoff).delete()
                db.commit()
        except Exception:
            logger.exception("Spatial: ошибка очистки старых наблюдений")

    # -------------------------------------------------------- внутреннее

    def _maybe_prune(self, now: float) -> None:
        if now - self._last_prune < PRUNE_INTERVAL:
            return
        self._last_prune = now
        # траектория живёт столько же, сколько identity в галерее:
        # переход между камерами может занять минуты — координаты
        # «где человек был» нужны скореру всё это время
        self.trajectories.prune(now, settings.global_gallery_ttl)
        # заодно чистим троттлинг-словарь
        with self._lock:
            if len(self._last_obs_write) > 512:
                self._last_obs_write.clear()

    def _persist_observation(self, global_id: int, state: TrajectorySnapshot,
                             floor_id: Optional[int]) -> None:
        from app.database.models import SpatialObservation
        try:
            with self._session_factory() as db:
                db.add(SpatialObservation(
                    global_id=global_id, camera_id=state.camera_id,
                    track_id=state.track_id, floor_id=floor_id,
                    world_x=state.position[0], world_y=state.position[1],
                    speed=state.speed, direction=state.direction,
                ))
                db.commit()
        except Exception:
            # FK на global_persons может отсутствовать, если person ещё не
            # записан — наблюдение просто пропускается
            logger.exception("Spatial: не записать наблюдение gid=%d",
                             global_id)
