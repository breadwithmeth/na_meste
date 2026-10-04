"""Распознавание действий человека: поза → геометрия → сглаживание.

Слой НАД существующим пайплайном (детекция/трекинг/лица не меняются):
раз в ACTION_INTERVAL секунд для каждого трека оценивается поза
(YOLOv8n-pose, 17 ключевых точек COCO), по геометрии скелета
классифицируется мгновенное действие, а по окну последних оценок —
итоговое (голосование + гистерезис, по образцу MIN_CONFIRMATIONS).

Мгновенные действия (правила по keypoints):
- lying   — торс ближе к горизонтали, чем к вертикали (лежит);
- phone   — кисть у уха (говорит по телефону);
- eating  — кисть у лица/рта (ест/пьёт);
- sitting — бёдра низко в боксе, колени у бёдер, бокс «квадратный» (сидит);
- walking — поза стоя + заметное перемещение (идёт);
- standing — всё остальное (стоит).

Производные действия (временные, на окне сглаживания):
- working — сидит/стоит с устойчиво занятыми руками в «рабочей зоне»
  перед корпусом (стол/прилавок) — работает;
- resting — сидит без активности рук дольше ACTION_REST_AFTER — отдыхает.

Результат пишется в DetectionResult (action/action_confidence/action_since),
в БД (action_observations, троттлинг + на смене действия) и виден в UI.
"""
import logging
import threading
import time
import urllib.request
from collections import deque
from pathlib import Path
from typing import Callable, Optional

import numpy as np

logger = logging.getLogger("app.ai.actions")

# --------------------------------------------------------------- ключи COCO
NOSE = 0
L_EAR, R_EAR = 3, 4
L_SHOULDER, R_SHOULDER = 5, 6
L_ELBOW, R_ELBOW = 7, 8
L_WRIST, R_WRIST = 9, 10
L_HIP, R_HIP = 11, 12
L_KNEE, R_KNEE = 13, 14
L_ANKLE, R_ANKLE = 15, 16

POSE_MODEL_URL = (
    "https://github.com/ultralytics/assets/releases/download/"
    "v8.4.0/yolov8n-pose.onnx"
)

ACTION_LABELS = {
    "standing": "стоит",
    "walking": "идёт",
    "sitting": "сидит",
    "lying": "лежит",
    "eating": "ест/пьёт",
    "phone": "телефон",
    "working": "работает",
    "resting": "отдыхает",
}

# пороговые константы классификатора (вынесены для тестов и настройки)
LYING_MAX_COS = 0.55        # cos угла торса от вертикали ниже — лежит
SITTING_MIN_SCORE = 1.5     # сумма признаков сидения
WALKING_SPEED = 0.7         # торс-длин в секунду
EAT_FACE_FRAC = 0.30        # доля кадров с рукой у лица → «ест/пьёт»
WORK_DESK_FRAC_SIT = 0.50   # доля кадров с руками в рабочей зоне (сидя)
WORK_DESK_FRAC_STAND = 0.60 # то же для стоящего человека


# --------------------------------------------------------- мгновенная поза

def _pt(kpts: np.ndarray, i: int, min_conf: float):
    """Точка (x, y), если уверенность достаточна, иначе None."""
    x, y, c = kpts[i]
    if c >= min_conf:
        return float(x), float(y)
    return None


def _mid(a, b):
    if a is None or b is None:
        return a or b  # одна из точек тоже годится как приближение центра
    return (a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0


def _dist(a, b) -> float:
    return float(np.hypot(a[0] - b[0], a[1] - b[1]))


def classify_pose(
    kpts: np.ndarray,
    bbox: tuple[float, float, float, float],
    speed: Optional[float] = None,
    min_conf: float = 0.30,
) -> tuple[str, float, str]:
    """Мгновенная классификация по одному скелету.

    kpts — (17, 3) [x, y, conf] в пикселях кадра; bbox — (x1, y1, x2, y2)
    детекции человека в тех же координатах; speed — скорость перемещения
    в торс-длинах/сек (None — неизвестна). Возвращает (action, confidence,
    hands), hands ∈ face | desk | down | none — положение рук для
    производных действий (working/resting).
    """
    x1, y1, x2, y2 = bbox
    bw, bh = max(1.0, x2 - x1), max(1.0, y2 - y1)

    sh_l = _pt(kpts, L_SHOULDER, min_conf)
    sh_r = _pt(kpts, R_SHOULDER, min_conf)
    hip_l = _pt(kpts, L_HIP, min_conf)
    hip_r = _pt(kpts, R_HIP, min_conf)
    shoulder_mid = _mid(sh_l, sh_r)
    hip_mid = _mid(hip_l, hip_r)
    nose = _pt(kpts, NOSE, min_conf)
    ear_l = _pt(kpts, L_EAR, min_conf)
    ear_r = _pt(kpts, R_EAR, min_conf)

    shoulder_w = _dist(sh_l, sh_r) if sh_l and sh_r else None
    torso_len = None
    if shoulder_mid and hip_mid:
        torso_len = _dist(shoulder_mid, hip_mid)
    if torso_len and torso_len < 0.5 * (shoulder_w or torso_len):
        torso_len = None  # подозрительно короткий торс — не доверяем

    # --- лежит: торс ближе к горизонтали ---
    if torso_len and shoulder_mid and hip_mid:
        cos_v = abs(hip_mid[1] - shoulder_mid[1]) / torso_len
        if cos_v < LYING_MAX_COS:
            return ("lying", round(min(0.9, 0.55 + (LYING_MAX_COS - cos_v)), 2),
                    _hands(kpts, shoulder_mid, hip_mid, torso_len, bbox, min_conf))

    # --- рука у лица: телефон (ухо) или еда (рот/нос) ---
    wrists = [w for w in (_pt(kpts, L_WRIST, min_conf), _pt(kpts, R_WRIST, min_conf))
              if w is not None]
    face_unit = _face_unit(ear_l, ear_r, nose, shoulder_mid, shoulder_w)
    if wrists and face_unit and nose:
        ears = [e for e in (ear_l, ear_r) if e is not None]
        near_ear = any(
            _dist(w, e) < 1.3 * face_unit and w[1] <= e[1] + 0.4 * face_unit
            for w in wrists for e in ears
        )
        near_mouth = any(
            _dist(w, nose) < 1.3 * face_unit and w[1] > nose[1] - 0.1 * face_unit
            for w in wrists
        )
        if near_ear:
            return "phone", 0.75, "face"
        if near_mouth:
            return "eating", 0.7, "face"

    # --- сидит: бёдра низко, колени у бёдер, бокс «квадратный» ---
    score = 0.0
    if hip_mid:
        hip_ratio = (hip_mid[1] - y1) / bh
        if hip_ratio >= 0.72:
            score += 1.0
        elif hip_ratio >= 0.64:
            score += 0.5
        elif hip_ratio <= 0.56:
            score -= 1.0  # бёдра высоко — точно стоит
    knee_mid = _mid(_pt(kpts, L_KNEE, min_conf), _pt(kpts, R_KNEE, min_conf))
    if knee_mid and hip_mid and torso_len:
        knee_gap = (knee_mid[1] - hip_mid[1]) / torso_len
        if knee_gap < 0.45:
            score += 1.0
        elif knee_gap < 0.75:
            score += 0.5
    aspect = bh / bw
    if aspect <= 1.55:
        score += 1.0
    elif aspect <= 1.85:
        score += 0.5
    if shoulder_mid:
        shoulder_ratio = (shoulder_mid[1] - y1) / bh
        if shoulder_ratio >= 0.42:
            score += 1.0
        elif shoulder_ratio >= 0.33:
            score += 0.5
    if score >= SITTING_MIN_SCORE:
        return ("sitting", round(min(0.85, 0.45 + 0.12 * score), 2),
                _hands(kpts, shoulder_mid, hip_mid, torso_len, bbox, min_conf))

    # --- идёт: перемещение заметнее порога ---
    if speed is not None and speed >= WALKING_SPEED:
        return "walking", round(min(0.9, 0.5 + speed * 0.15), 2), "down"

    return "standing", 0.5, _hands(
        kpts, shoulder_mid, hip_mid, torso_len, bbox, min_conf)


def _face_unit(ear_l, ear_r, nose, shoulder_mid, shoulder_w) -> Optional[float]:
    """Масштаб головы: расстояние между ушами (или его приближения)."""
    if ear_l and ear_r:
        return _dist(ear_l, ear_r)
    ear = ear_l or ear_r
    if ear and nose:
        return 1.6 * _dist(ear, nose)
    if nose and shoulder_mid:
        return 0.5 * max(_dist(nose, shoulder_mid), shoulder_w or 0.0)
    if shoulder_w:
        return 0.5 * shoulder_w
    return None


def _hands(kpts, shoulder_mid, hip_mid, torso_len, bbox, min_conf) -> str:
    """Положение рук: face — у лица, desk — в «рабочей зоне» перед корпусом
    (между плечами и бёдрами: стол/прилавок), down — опущены/не видны."""
    wrists = [w for w in (_pt(kpts, L_WRIST, min_conf), _pt(kpts, R_WRIST, min_conf))
              if w is not None]
    if not wrists or not shoulder_mid:
        return "none"
    _, y1, _, y2 = bbox
    for w in wrists:
        if hip_mid and torso_len:
            if shoulder_mid[1] + 0.1 * torso_len < w[1] < hip_mid[1]:
                return "desk"
        else:
            # бёдра скрыты (например, стол): рабочая зона — нижняя часть
            # видимого бокса ниже плеч
            if w[1] > shoulder_mid[1] + 0.6 * (y2 - shoulder_mid[1]):
                return "desk"
    return "down"


# ------------------------------------------------------- состояние трека

class _TrackState:
    """Всё, что помнит классификатор о треке: голоса, скорость, действие."""

    __slots__ = (
        "votes", "anchors", "speed_ema", "last_pose_ts", "updated",
        "base_action", "final_action", "final_conf", "final_since",
        "last_hands_activity", "last_emit", "last_emit_ids",
    )

    def __init__(self, now: float):
        self.votes: deque = deque()          # (ts, action, hands)
        self.anchors: deque = deque()        # (ts, x, y) — центр тела
        self.speed_ema: Optional[float] = None
        self.last_pose_ts = 0.0
        self.updated = now
        self.base_action: Optional[str] = None   # большинство мгновенных
        self.final_action: Optional[str] = None  # итоговая метка для UI/БД
        self.final_conf: Optional[float] = None
        self.final_since = now                # monotonic — начало действия
        self.last_hands_activity = now        # руки были у лица/в рабочей зоне
        self.last_emit = 0.0                  # последняя запись в БД
        self.last_emit_ids: Optional[tuple] = None  # (employee_id, global_id)


# ---------------------------------------------------------- распознаватель

class ActionRecognizer:
    """Обогащает DetectionResult действием человека. Потокобезопасен
    (на случай нескольких AI-воркеров), вызывается из воркера."""

    STATE_TTL = 120.0  # сек без обновлений → состояние трека удаляется

    def __init__(
        self,
        pose,
        on_action: Callable,
        *,
        interval: float = 1.0,
        smooth_seconds: float = 6.0,
        min_conf: float = 0.30,
        rest_after: float = 15.0,
        log_interval: float = 10.0,
    ):
        self.pose = pose
        self.on_action = on_action      # (camera_id, track_id, employee_id,
                                        #  global_id, action, confidence)
        self.interval = interval
        self.smooth_seconds = smooth_seconds
        self.min_conf = min_conf
        self.rest_after = rest_after
        self.log_interval = log_interval
        self._states: dict[tuple[int, int], _TrackState] = {}
        self._lock = threading.Lock()

    # ------------------------------------------------------------- публичное

    def process(self, camera_id: int, detections, frame: np.ndarray,
                now: float) -> None:
        """Оценить позы (с троттлингом на трек), сгладить и проставить
        action/action_confidence/action_since в каждый DetectionResult;
        смены и периодические наблюдения отправить в on_action."""
        h, w = frame.shape[:2]
        records: list[tuple] = []
        with self._lock:
            for det in detections:
                key = (camera_id, det.track_id)
                st = self._states.get(key)
                if st is None:
                    st = self._states[key] = _TrackState(now)
                st.updated = now

                if now - st.last_pose_ts >= self.interval:
                    st.last_pose_ts = now
                    self._estimate(st, det, frame, w, h, now)

                changed = self._smooth(st, now)
                det.action = st.final_action
                det.action_confidence = st.final_conf
                if st.final_action is not None:
                    det.action_since = time.time() - (now - st.final_since)

                ids = (det.employee_id, det.global_id)
                if st.final_action is not None and (
                    changed
                    or st.last_emit_ids != ids
                    or now - st.last_emit >= self.log_interval
                ):
                    st.last_emit = now
                    st.last_emit_ids = ids
                    records.append((
                        camera_id, det.track_id, det.employee_id,
                        det.global_id, st.final_action, st.final_conf,
                    ))

            self._prune(now)
        for record in records:
            try:
                self.on_action(*record)
            except Exception:
                logger.exception("Не удалось записать наблюдение действия")

    def drop_camera(self, camera_id: int) -> None:
        with self._lock:
            for key in [k for k in self._states if k[0] == camera_id]:
                self._states.pop(key, None)

    # ------------------------------------------------------------- внутреннее

    def _estimate(self, st: _TrackState, det, frame, w, h, now: float) -> None:
        """Кроп человека → поза → скорость + голос."""
        x1, y1, x2, y2 = det.bbox  # нормализованные координаты 0..1
        px1, py1 = max(0, int(x1 * w)), max(0, int(y1 * h))
        px2, py2 = min(w, int(x2 * w)), min(h, int(y2 * h))
        if px2 - px1 < 32 or py2 - py1 < 48:
            return  # человек слишком мал для позы на этом потоке
        mx, my = int((px2 - px1) * 0.15), int((py2 - py1) * 0.15)
        cx1, cy1 = max(0, px1 - mx), max(0, py1 - my)
        cx2, cy2 = min(w, px2 + mx), min(h, py2 + my)
        crop = frame[cy1:cy2, cx1:cx2]
        kpts = self.pose.estimate(crop)
        if kpts is None:
            return
        kpts[:, 0] += cx1
        kpts[:, 1] += cy1

        self._update_speed(st, kpts, now)
        if st.votes and now - st.votes[-1][0] > self.smooth_seconds:
            st.votes.clear()           # разрыв в наблюдениях (окклюзия) —
            st.last_hands_activity = now  # история голосов неактуальна
        action, conf, hands = classify_pose(
            kpts, (px1, py1, px2, py2), st.speed_ema, self.min_conf)
        st.votes.append((now, action, hands))
        if hands in ("face", "desk"):
            st.last_hands_activity = now
        while st.votes and now - st.votes[0][0] > self.smooth_seconds:
            st.votes.popleft()

    def _update_speed(self, st: _TrackState, kpts: np.ndarray, now: float) -> None:
        """Скорость центра тела в торс-длинах/сек (EMA)."""
        hip_mid = _mid(_pt(kpts, L_HIP, self.min_conf),
                       _pt(kpts, R_HIP, self.min_conf))
        shoulder_mid = _mid(_pt(kpts, L_SHOULDER, self.min_conf),
                            _pt(kpts, R_SHOULDER, self.min_conf))
        center = hip_mid or shoulder_mid
        if center is None:
            return
        scale = None
        if hip_mid and shoulder_mid:
            scale = _dist(hip_mid, shoulder_mid)
        if scale is None or scale < 8:  # слишком мелко — скорость не считаем
            st.anchors.clear()
            st.speed_ema = None
            st.anchors.append((now, *center))
            return
        if st.anchors:
            ts, ax, ay = st.anchors[-1]
            dt = now - ts
            if dt > 3.0:
                st.speed_ema = None  # трек пропадал — историю не учитываем
            elif dt > 0.05:
                v = _dist(center, (ax, ay)) / dt / scale
                st.speed_ema = v if st.speed_ema is None else \
                    0.6 * st.speed_ema + 0.4 * v
        st.anchors.append((now, *center))
        if len(st.anchors) > 8:
            st.anchors.popleft()

    def _smooth(self, st: _TrackState, now: float) -> bool:
        """Голосование по окну + производные действия. True — действие сменилось."""
        while st.votes and now - st.votes[0][0] > self.smooth_seconds:
            st.votes.popleft()
        window = list(st.votes)
        if not window:
            st.base_action = None
            if st.final_action is not None:
                st.final_action = None
                st.final_conf = None
                return True
            return False

        counts: dict[str, int] = {}
        for _, action, _ in window:
            counts[action] = counts.get(action, 0) + 1
        best, best_n = max(counts.items(), key=lambda kv: kv[1])
        if st.base_action is None or best != st.base_action:
            # гистерезис: новое действие должно строго перевесить текущее
            if counts.get(st.base_action, 0) < best_n:
                st.base_action = best

        n = len(window)
        eat_frac = sum(1 for _, a, _ in window if a == "eating") / n
        phone_frac = sum(1 for _, a, _ in window if a == "phone") / n
        desk_frac = sum(1 for _, _, h in window if h == "desk") / n

        base = st.base_action
        final, conf = base, max(0.4, best_n / n)
        if base == "sitting":
            if eat_frac >= EAT_FACE_FRAC and eat_frac >= phone_frac:
                final, conf = "eating", min(0.9, eat_frac + 0.35)
            elif phone_frac >= EAT_FACE_FRAC:
                final, conf = "phone", min(0.9, phone_frac + 0.35)
            elif desk_frac >= WORK_DESK_FRAC_SIT:
                final, conf = "working", min(0.9, desk_frac + 0.3)
            elif now - st.last_hands_activity >= self.rest_after:
                final, conf = "resting", 0.6
        elif base == "standing":
            if desk_frac >= WORK_DESK_FRAC_STAND:
                final, conf = "working", min(0.9, desk_frac + 0.3)

        changed = final != st.final_action
        if changed:
            st.final_since = now
        st.final_action = final
        st.final_conf = round(conf, 2)
        return changed

    def _prune(self, now: float) -> None:
        stale = [k for k, v in self._states.items()
                 if now - v.updated > self.STATE_TTL]
        for key in stale:
            self._states.pop(key, None)


# ------------------------------------------------------------ загрузка модели

def download_pose_model(dest: Path, url: str = POSE_MODEL_URL) -> bool:
    """Скачать pose-модель при первом запуске (один раз, ~13 МБ)."""
    try:
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_suffix(".part")
        logger.info("Скачиваю pose-модель: %s", url)
        with urllib.request.urlopen(url, timeout=60) as resp, open(tmp, "wb") as f:
            while True:
                chunk = resp.read(1 << 20)
                if not chunk:
                    break
                f.write(chunk)
        tmp.replace(dest)
        logger.info("Pose-модель сохранена: %s", dest)
        return True
    except Exception:
        logger.exception("Не удалось скачать pose-модель (%s)", url)
        try:
            dest.with_suffix(".part").unlink(missing_ok=True)
        except OSError:
            pass
        return False
