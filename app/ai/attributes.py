"""Атрибуты внешности человека: цвет одежды (верх/низ) по кропу бокса.

Слой НАД существующим пайплайном: без моделей — сэмплируются полосы
person-бокса (торс, ноги), цвет называется по HSV-статистике и показывается
только после нескольких стабильных сэмплов (антифликер).

- торс  = полоса y 0.15–0.45 бокса (голова не попадает);
- ноги  = полоса y 0.60–0.92 бокса.

Итог: название цвета («синий», «чёрный»…) + hex для свотча в UI пишутся
в DetectionResult (upper_color/upper_hex/lower_color/lower_hex), а вместе
с полом/возрастом (genderage, см. RecognitionService) — троттлингом
в БД (appearance_observations) через on_attributes.
"""
import logging
import threading
from collections import Counter

import cv2
import numpy as np

logger = logging.getLogger("app.ai.attributes")

# пороги HSV (OpenCV: H 0..179, S/V 0..255)
BLACK_MAX_V = 50        # V ниже — чёрный
GRAY_MAX_S = 42         # S ниже — белый/серый (ахроматические)
WHITE_MIN_V = 200       # ахроматический и яркий — белый
BROWN_MAX_V = 135       # оранжево-красный и тёмный — коричневый

# hue-диапазоны хроматических цветов (границы в OpenCV hue)
HUE_BUCKETS = (
    (5, 22, "оранжевый"),
    (22, 33, "жёлтый"),
    (33, 78, "зелёный"),
    (78, 99, "голубой"),
    (99, 130, "синий"),
    (130, 150, "фиолетовый"),
    (150, 170, "розовый"),
)
# красный — «заворачивается» вокруг нуля: H < 5 или H >= 170

STABLE_SAMPLES = 3      # сэмплов одного цвета в окне до показа
WINDOW = 6              # размер окна голосования (сэмплов)


def name_color(patch_bgr: np.ndarray) -> tuple[str, str] | None:
    """Патч BGR → (название цвета, hex медианного цвета). None — пусто."""
    if patch_bgr is None or patch_bgr.size == 0:
        return None
    hsv = cv2.cvtColor(patch_bgr, cv2.COLOR_BGR2HSV)
    h = float(np.median(hsv[:, :, 0]))
    s = float(np.median(hsv[:, :, 1]))
    v = float(np.median(hsv[:, :, 2]))

    med = np.median(patch_bgr.reshape(-1, 3), axis=0).astype(int)
    hex_color = "#{:02X}{:02X}{:02X}".format(int(med[2]), int(med[1]), int(med[0]))

    if v < BLACK_MAX_V:
        return "чёрный", hex_color
    if s < GRAY_MAX_S:
        return ("белый", hex_color) if v >= WHITE_MIN_V else ("серый", hex_color)
    if h < 22 or h >= 170:
        # красно-оранжевый сектор: тёмные оттенки — коричневый
        if v < BROWN_MAX_V:
            return "коричневый", hex_color
        return ("красный" if (h < 5 or h >= 170) else "оранжевый"), hex_color
    for lo, hi, name in HUE_BUCKETS:
        if lo <= h < hi:
            return name, hex_color
    return "красный", hex_color


def _band(frame: np.ndarray, bbox, y_lo: float, y_hi: float) -> np.ndarray | None:
    """Полоса person-бокса (норм. координаты) → пиксельный патч."""
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = bbox
    px1, px2 = max(0, int(x1 * w)), min(w, int(x2 * w))
    py1 = max(0, int((y1 + (y2 - y1) * y_lo) * h))
    py2 = min(h, int((y1 + (y2 - y1) * y_hi) * h))
    if px2 - px1 < 12 or py2 - py1 < 12:
        return None  # человек слишком мал — цвет недостоверен
    return frame[py1:py2, px1:px2]


class _TrackState:
    """Голоса цветов по треку + отметки записи в БД."""
    __slots__ = ("upper_votes", "lower_votes", "last_sample",
                 "last_log", "last_logged")

    def __init__(self):
        self.upper_votes: list[tuple[str, str]] = []   # (название, hex)
        self.lower_votes: list[tuple[str, str]] = []
        self.last_sample = 0.0
        self.last_log = 0.0
        self.last_logged: tuple | None = None


class AttributeExtractor:
    """Обогащает DetectionResult цветами одежды и пишет наблюдения в БД.
    Потокобезопасен (на случай нескольких AI-воркеров)."""

    STATE_TTL = 120.0  # сек без обновлений → состояние трека удаляется

    def __init__(self, on_attributes=None, interval: float = 0.5,
                 log_interval: float = 30.0):
        self.on_attributes = on_attributes
        self.interval = interval
        self.log_interval = log_interval
        self._states: dict[tuple[int, int], _TrackState] = {}
        self._lock = threading.Lock()

    def process(self, camera_id: int, detections, frame, now: float) -> None:
        records: list[tuple] = []
        with self._lock:
            for det in detections:
                key = (camera_id, det.track_id)
                st = self._states.get(key)
                if st is None:
                    st = self._states[key] = _TrackState()

                if now - st.last_sample >= self.interval:
                    st.last_sample = now
                    self._sample(st, det, frame)

                # стабилизированные цвета — в результат
                upper = self._stable(st.upper_votes)
                lower = self._stable(st.lower_votes)
                if upper:
                    det.upper_color, det.upper_hex = upper
                if lower:
                    det.lower_color, det.lower_hex = lower

                # запись в БД: при смене облика или с троттлингом
                attrs = self._snapshot(det)
                if attrs is not None and (
                    attrs != st.last_logged
                    or now - st.last_log >= self.log_interval
                ):
                    st.last_logged = attrs
                    st.last_log = now
                    records.append((camera_id, det.track_id, det.employee_id,
                                    det.global_id) + attrs)

            self._prune(now)
        for record in records:
            try:
                self.on_attributes(*record)
            except Exception:
                logger.exception("Не удалось записать наблюдение внешности")

    def drop_camera(self, camera_id: int) -> None:
        with self._lock:
            for key in [k for k in self._states if k[0] == camera_id]:
                self._states.pop(key, None)

    # ------------------------------------------------------------ внутреннее

    def _sample(self, st: _TrackState, det, frame: np.ndarray) -> None:
        """Один сэмпл цветов торса/ног (стабилизация — голосованием)."""
        upper = name_color(_band(frame, det.bbox, 0.15, 0.45))
        lower = name_color(_band(frame, det.bbox, 0.60, 0.92))
        if upper:
            st.upper_votes.append(upper)
        if lower:
            st.lower_votes.append(lower)
        del st.upper_votes[:-WINDOW]
        del st.lower_votes[:-WINDOW]

    @staticmethod
    def _stable(votes: list[tuple[str, str]]) -> tuple[str, str] | None:
        """Цвет показываем, если в окне он набрал STABLE_SAMPLES голосов;
        hex — от последнего голоса этого цвета."""
        if len(votes) < STABLE_SAMPLES:
            return None
        counts = Counter(name for name, _ in votes)
        name, count = counts.most_common(1)[0]
        if count < STABLE_SAMPLES:
            return None
        hex_color = next(hexv for vname, hexv in reversed(votes) if vname == name)
        return name, hex_color

    @staticmethod
    def _snapshot(det) -> tuple | None:
        """(upper, lower, gender, age) для БД; None — пока нечего писать."""
        if det.upper_color is None and det.lower_color is None \
                and det.gender is None and det.age is None:
            return None
        return (det.upper_color, det.lower_color, det.gender, det.age)

    def _prune(self, now: float) -> None:
        stale = [k for k, v in self._states.items()
                 if now - v.last_sample > self.STATE_TTL]
        for key in stale:
            self._states.pop(key, None)
