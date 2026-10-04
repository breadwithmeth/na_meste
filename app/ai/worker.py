"""AI-воркер: один поток обрабатывает кадры всех камер.

Архитектура (ТЗ «Несколько камер → один AI-воркер»):

    Camera 1..N ─► FrameBuffer ридеров (queue=2, drop old / keep latest)
                        │
                        ▼
                   AI Worker ──► RecognitionService ──► DetectionState (для UI/API)
                        │
                        ▼
                 PresenceManager (сессии присутствия)

Кадры берутся напрямую из FrameBuffer ридеров: буфер уже реализует
«max queue size = 2, старый кадр отбрасывается» — бесконечных очередей нет.
Не успеваем — камера просто обрабатывается реже, превью не страдает.
Для масштабирования можно запустить несколько воркеров с разбиением по камерам
(service потокобезопасен).
"""
import logging
import threading
import time
from typing import Optional

from app.ai.detector import PersonDetector
from app.ai.face_detector import FaceEngine
from app.ai.face_recognition import EmbeddingStore
from app.ai.providers import provider_label, resolve_providers, setup_cuda_dlls
from app.ai.recognition_service import RecognitionService
from app.config import BASE_DIR, settings
from app.rtsp.manager import ReaderManager
from app.services.presence_service import PresenceManager

logger = logging.getLogger("app.ai.worker")


class DetectionState:
    """Потокобезопасные последние результаты детекции по камерам (для API/UI)."""

    def __init__(self):
        self._lock = threading.Lock()
        self._data: dict[int, dict] = {}

    def update(self, camera_id: int, detections) -> None:
        tracks = [
            {
                "track_id": d.track_id,
                "employee_id": d.employee_id,
                "employee_name": d.employee_name,
                "confidence": d.confidence,
                "bbox": list(d.bbox),
                "state": d.state,
                "global_id": d.global_id,
            }
            for d in detections
        ]
        with self._lock:
            self._data[camera_id] = {
                "camera_id": camera_id,
                "updated_at": time.time(),
                "tracks": tracks,
                "people_count": len(tracks),
                "known_count": sum(1 for t in tracks if t["employee_id"]),
                "unknown_count": sum(1 for t in tracks if not t["employee_id"]),
            }

    def camera(self, camera_id: int) -> Optional[dict]:
        with self._lock:
            return self._data.get(camera_id)

    def counts(self, camera_id: int) -> tuple[int, int, int]:
        with self._lock:
            item = self._data.get(camera_id)
            if not item:
                return 0, 0, 0
            return item["people_count"], item["known_count"], item["unknown_count"]

    def drop_camera(self, camera_id: int) -> None:
        with self._lock:
            self._data.pop(camera_id, None)


class AIWorker(threading.Thread):
    def __init__(
        self,
        manager: ReaderManager,
        service: RecognitionService,
        presence: PresenceManager,
        detection_state: DetectionState,
        unknown_manager=None,
        global_manager=None,
    ):
        super().__init__(name="ai-worker", daemon=True)
        self.manager = manager
        self.service = service
        self.presence = presence
        self.state = detection_state
        self.unknown_manager = unknown_manager
        self.global_manager = global_manager
        self._stop = threading.Event()
        self._last_seq: dict[int, int] = {}
        self._last_ts: dict[int, float] = {}
        self._last_cleanup = 0.0
        self._last_unknown: dict[int, float] = {}
        self._last_unknown_cleanup = 0.0
        self._last_global_cleanup = 0.0

    def stop(self, timeout: float = 3.0) -> None:
        self._stop.set()
        self.join(timeout=timeout)

    def run(self) -> None:
        logger.info(
            "AI worker запущен (AI_FPS=%.1f на камеру)", settings.ai_fps
        )
        interval = 1.0 / max(0.5, settings.ai_fps)

        while not self._stop.is_set():
            did_work = False
            now = time.monotonic()

            for camera_id in self._camera_ids():
                if self._stop.is_set():
                    break
                reader = self.manager.get(camera_id)
                if reader is None or not reader.is_alive():
                    self.state.drop_camera(camera_id)
                    continue
                item = reader.buffer.latest()
                if item is None:
                    continue
                seq, frame = item
                if seq <= self._last_seq.get(camera_id, 0):
                    continue  # нового кадра нет
                if now - self._last_ts.get(camera_id, 0.0) < interval:
                    continue  # ещё рано для этой камеры (лимит AI_FPS)
                self._last_seq[camera_id] = seq
                self._last_ts[camera_id] = now
                try:
                    outcome = self.service.process_frame(frame, camera_id, now)
                except Exception:
                    logger.exception("Camera %d: ошибка AI-обработки кадра", camera_id)
                    continue

                # межкамерный трекинг: Re-ID + глобальные identity.
                # Ошибки этого слоя не должны останавливать pipeline —
                # detection/tracking/presence продолжают работать.
                if self.global_manager is not None:
                    try:
                        self.global_manager.process(camera_id, outcome, frame, now)
                    except Exception:
                        logger.exception(
                            "Camera %d: ошибка глобального трекинга — "
                            "локальный трекинг продолжается", camera_id,
                        )

                self.state.update(camera_id, outcome.detections)
                for _track_id, employee_id, confidence in outcome.new_recognitions:
                    self.presence.on_recognized(camera_id, employee_id, confidence)
                # фиксация посторонних (с cooldown на камеру, чтобы трек-чёрн
                # одного человека не заспамил события); фиксация связывается
                # с глобальной личностью человека, если трекинг включён
                if outcome.unknown_events and self.unknown_manager is not None:
                    if (now - self._last_unknown.get(camera_id, 0.0)
                            >= settings.unknown_event_cooldown):
                        self._last_unknown[camera_id] = now
                        track_id, bbox = outcome.unknown_events[0]
                        global_id = next(
                            (d.global_id for d in outcome.detections
                             if d.track_id == track_id), None)
                        self.unknown_manager.record(
                            camera_id, track_id, frame, bbox, global_id=global_id)
                did_work = True

            # закрыть сессии присутствия, истёкшие по таймауту
            self.presence.tick()

            # подчистить состояние исчезнувших камер
            if now - self._last_cleanup > 10.0:
                self._last_cleanup = now
                for camera_id in list(self._last_seq):
                    if self.manager.get(camera_id) is None:
                        self._last_seq.pop(camera_id, None)
                        self._last_ts.pop(camera_id, None)
                        self._last_unknown.pop(camera_id, None)
                        self.service.drop_camera(camera_id)
                        self.state.drop_camera(camera_id)
                # раз в час — удалить старые фиксации посторонних и глобальные данные
                if self.unknown_manager is not None and now - self._last_unknown_cleanup > 3600.0:
                    self._last_unknown_cleanup = now
                    self.unknown_manager.cleanup()
                if (self.global_manager is not None
                        and now - self._last_global_cleanup > 3600.0):
                    self._last_global_cleanup = now
                    self.global_manager.cleanup()

            if not did_work:
                self._stop.wait(0.05)

        logger.info("AI worker остановлен")

    def _camera_ids(self) -> list[int]:
        ids = set(self.manager.reader_ids())
        allowed = settings.ai_camera_ids()
        if allowed:
            ids &= allowed
        return sorted(ids)


# ------------------------------------------------------------- инициализация

def build_enroll_engine() -> Optional[FaceEngine]:
    """Движок для загрузки фото сотрудников (det_size 640). None — AI недоступен."""
    setup_cuda_dlls()
    try:
        return FaceEngine(
            model_name=settings.insightface_model,
            root=settings.insightface_root,
            providers=resolve_providers(settings.ai_device),
            det_size=640,
        )
    except Exception as exc:
        logger.error("FaceEngine недоступна (загрузка фото отключена): %s", exc)
        return None


def build_ai_worker(
    manager: ReaderManager, presence: PresenceManager, store: EmbeddingStore,
    unknown_manager=None, session_factory=None, notifier=None,
) -> Optional[AIWorker]:
    """Собирает полный AI-стек. None — модели недоступны (приложение работает
    только как RTSP-превью, не падая)."""
    setup_cuda_dlls()
    providers = resolve_providers(settings.ai_device)

    yolo_path = BASE_DIR / settings.yolo_model
    if not yolo_path.is_file():
        logger.error("YOLO-модель не найдена: %s — AI отключён (см. README)", yolo_path)
        return None

    # межкамерный трекинг (опциональный слой, не ломает остальное)
    global_manager = None
    if settings.multi_camera_tracking_enabled and session_factory is not None:
        reid_path = BASE_DIR / settings.reid_model
        if reid_path.is_file():
            try:
                from app.ai.global_tracker import CameraTopology, GlobalIdentityManager
                from app.ai.reid import PersonReID

                reid = PersonReID(str(reid_path), providers)
                topology = CameraTopology.load(settings.topology_config)
                global_manager = GlobalIdentityManager(
                    session_factory, reid, topology, notifier=notifier
                )
                logger.info("Межкамерный трекинг включён (Re-ID: %s)",
                            reid_path.name)
            except Exception:
                logger.exception(
                    "Межкамерный трекинг не запустился — система работает без него")
                global_manager = None
        else:
            logger.warning("Re-ID модель не найдена: %s — межкамерный трекинг "
                           "отключён (см. README)", reid_path)

    try:
        detector = PersonDetector(str(yolo_path), providers, settings.person_confidence)
        face_engine = FaceEngine(
            model_name=settings.insightface_model,
            root=settings.insightface_root,
            providers=providers,
            det_size=320,
        )
    except Exception as exc:
        logger.error("AI-модели не загрузились (%s) — AI отключён", exc)
        return None

    service = RecognitionService(
        detector,
        face_engine,
        store,
        min_confidence=settings.min_recognition_confidence,
        min_confirmations=settings.min_confirmations,
        retry_interval=settings.face_recognition_retry_interval,
        track_lost_timeout=settings.track_lost_timeout,
    )
    state = DetectionState()
    logger.info("AI Provider: %s", provider_label(detector.session.get_providers()))
    return AIWorker(manager, service, presence, state,
                    unknown_manager=unknown_manager,
                    global_manager=global_manager)
