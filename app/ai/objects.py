"""Распознавание предметов сцены: еда на столе, посуда, оружие, носимые вещи.

Слой НАД существующим пайплайном (детекция/трекинг/лица не меняются).
Сырые детекции предметов приходят из той же YOLO-проходки, что и люди
(FrameOutcome.objects), здесь они:

- сглаживаются и «доживают» между кадрами (IoU-матчинг по метке, EMA бокса,
  TTL, показ после OBJECT_MIN_HITS подряд — антифликер);
- привязываются к людям: центр объекта внутри person-бокса (или IoU ≥
  PERSON_IOU) → треку в DetectionResult.items добавляется метка
  («нож», «рюкзак», «чашка»…);
- при подтверждении объекта (MIN_HITS подряд) вызывается on_object —
  запись в БД + Telegram-алерт для оружия (ObjectEventManager).

Порог уверенности и список классов настраиваются детектором (см. config),
здесь только категории и русские подписи.
"""
import logging
import threading
from dataclasses import dataclass, field

import numpy as np

logger = logging.getLogger("app.ai.objects")

# ------------------------------------------------------------ категории
WEAPON_LABELS = {"knife", "scissors"}
FOOD_LABELS = {
    "banana", "apple", "sandwich", "orange", "broccoli", "carrot",
    "hot dog", "pizza", "donut", "cake",
}
TABLEWARE_LABELS = {"bottle", "wine glass", "cup", "fork", "spoon", "bowl"}
CARRIED_LABELS = {
    "backpack", "handbag", "suitcase", "umbrella", "cell phone", "laptop",
}
FURNITURE_LABELS = {"dining table", "chair", "couch", "bed"}

CATEGORY_LABELS = {
    "weapon": "оружие",
    "food": "еда",
    "tableware": "посуда",
    "carried": "носимые вещи",
    "furniture": "мебель",
    "other": "прочее",
}


def category_of(label: str) -> str:
    if label in WEAPON_LABELS:
        return "weapon"
    if label in FOOD_LABELS:
        return "food"
    if label in TABLEWARE_LABELS:
        return "tableware"
    if label in CARRIED_LABELS:
        return "carried"
    if label in FURNITURE_LABELS:
        return "furniture"
    return "other"


# Русские подписи включённых классов (остальные — английское имя как есть)
COCO_RU = {
    "knife": "нож", "scissors": "ножницы", "fork": "вилка", "spoon": "ложка",
    "bowl": "миска", "cup": "чашка", "wine glass": "бокал", "bottle": "бутылка",
    "banana": "банан", "apple": "яблоко", "sandwich": "сэндвич",
    "orange": "апельсин", "broccoli": "брокколи", "carrot": "морковь",
    "hot dog": "хот-дог", "pizza": "пицца", "donut": "пончик", "cake": "торт",
    "dining table": "обеденный стол", "backpack": "рюкзак", "handbag": "сумка",
    "suitcase": "чемодан", "umbrella": "зонт", "cell phone": "телефон",
    "laptop": "ноутбук", "chair": "стул", "couch": "диван", "bed": "кровать",
    "book": "книга", "clock": "часы", "vase": "ваза",
    "teddy bear": "игрушка", "remote": "пульт", "keyboard": "клавиатура",
    "mouse": "мышь", "tie": "галстук", "potted plant": "растение",
    "microwave": "микроволновка", "oven": "печь", "sink": "раковина",
    "refrigerator": "холодильник", "tv": "телевизор", "toilet": "унитаз",
    # типовые классы сторонних моделей (EXTRA_MODELS / YOLO-World)
    "chips bag": "чипсы", "snack bag": "снек", "snack": "снек",
    "cookies": "печенье", "cookie": "печенье", "candy bar": "шоколадка",
    "chocolate bar": "шоколадка", "soda can": "банка газировки",
    "juice box": "сок в пакете", "coffee cup": "стакан кофе",
    "cigarette": "сигарета", "lighter": "зажигалка", "wallet": "кошелёк",
}


def ru_label(name: str) -> str:
    return COCO_RU.get(name, name)


@dataclass
class SceneObject:
    """Предмет на кадре для UI/БД (bbox нормализован 0..1)."""
    label: str
    label_ru: str
    category: str
    confidence: float
    bbox: tuple[float, float, float, float]
    track_ids: list[int] = field(default_factory=list)


def iou(a, b) -> float:
    x1 = max(a[0], b[0]); y1 = max(a[1], b[1])
    x2 = min(a[2], b[2]); y2 = min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if inter <= 0:
        return 0.0
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    return inter / (area_a + area_b - inter)


def _contains(box, point) -> bool:
    return box[0] <= point[0] <= box[2] and box[1] <= point[1] <= box[3]


class _ObjState:
    """Всё, что помнит слой об одном предмете камеры."""
    __slots__ = ("label", "label_ru", "category", "bbox", "score",
                 "hits", "first_seen", "last_seen", "track_ids")

    def __init__(self, label: str, bbox, score: float, now: float):
        self.label = label
        self.label_ru = ru_label(label)
        self.category = category_of(label)
        self.bbox = np.asarray(bbox, dtype=float)
        self.score = score
        self.hits = 1
        self.first_seen = now
        self.last_seen = now
        self.track_ids: list[int] = []


class SceneObjectTracker:
    """Обогащает DetectionResult носимыми предметами и отдаёт объекты сцены
    для UI/БД. Потокобезопасен (на случай нескольких AI-воркеров)."""

    STATE_TTL = 2.5       # сек жизни объекта без подтверждений
    MIN_HITS = 2          # подтверждений до показа/алерта (антифликер)
    MATCH_IOU = 0.30      # IoU матчинга сырой детекции с объектом
    PERSON_IOU = 0.15     # IoU объекта с человеком для привязки «при нём»
    MAX_ITEMS = 3         # максимум меток-предметов на человеке

    def __init__(self, on_object=None):
        # on_object(camera_id, SceneObject, frame, person|None) — вызывается
        # один раз при подтверждении объекта; кулдауны/алерты — в менеджере
        self.on_object = on_object
        self._states: dict[int, list[_ObjState]] = {}
        self._lock = threading.Lock()

    def process(self, camera_id: int, raw_objects, detections, frame, now: float):
        """raw_objects — SceneObject-сырье без сглаживания (build_scene_objects);
        detections — DetectionResult этого кадра; возвращает подтверждённые
        SceneObject (для DetectionState), проставляет det.items."""
        h, w = frame.shape[:2]
        with self._lock:
            states = self._states.setdefault(camera_id, [])
            matched: set[int] = set()
            for raw in raw_objects:
                best, best_iou = None, self.MATCH_IOU
                for idx, st in enumerate(states):
                    if st.label != raw.label or idx in matched:
                        continue
                    v = iou(st.bbox, raw.bbox)
                    if v >= best_iou:
                        best, best_iou = idx, v
                if best is None:
                    states.append(_ObjState(
                        raw.label, raw.bbox, raw.confidence, now))
                else:
                    st = states[best]
                    st.bbox = 0.6 * st.bbox + 0.4 * np.asarray(raw.bbox, dtype=float)
                    st.score = 0.6 * st.score + 0.4 * raw.confidence
                    st.hits += 1
                    st.last_seen = now
                    matched.add(best)

            # протухшие объекты (пропали из кадра) — убрать
            self._states[camera_id] = states = [
                st for st in states if now - st.last_seen <= self.STATE_TTL
            ]

            confirmed = [st for st in states if st.hits >= self.MIN_HITS]

            # привязка предметов к людям: центр объекта (в пикселях) внутри
            # person-бокса или IoU
            for st in confirmed:
                center = ((st.bbox[0] + st.bbox[2]) / 2.0 * w,
                          (st.bbox[1] + st.bbox[3]) / 2.0 * h)
                st.track_ids = []
                for det in detections:
                    box = (det.bbox[0] * w, det.bbox[1] * h,
                           det.bbox[2] * w, det.bbox[3] * h)
                    if _contains(box, center) or iou(box, st.bbox) >= self.PERSON_IOU:
                        st.track_ids.append(det.track_id)
                        if st.label_ru not in det.items:
                            det.items.append(st.label_ru)
                            if len(det.items) > self.MAX_ITEMS:
                                del det.items[self.MAX_ITEMS:]

            # событие — один раз на объект, в момент подтверждения (после
            # привязки к людям); кулдаун повторных появлений — в менеджере
            events = [
                (st, self._person_of(detections, st))
                for st in states if st.hits == self.MIN_HITS
            ]
        for st, person in events:
            if self.on_object is not None:
                try:
                    self.on_object(camera_id, SceneObject(
                        st.label, st.label_ru, st.category, round(st.score, 3),
                        (round(float(st.bbox[0]), 4), round(float(st.bbox[1]), 4),
                         round(float(st.bbox[2]), 4), round(float(st.bbox[3]), 4)),
                        list(st.track_ids)), frame, person)
                except Exception:
                    logger.exception("Camera %d: обработка события предмета", camera_id)
        return [
            SceneObject(st.label, st.label_ru, st.category,
                        round(st.score, 3),
                        (round(float(st.bbox[0]), 4), round(float(st.bbox[1]), 4),
                         round(float(st.bbox[2]), 4), round(float(st.bbox[3]), 4)),
                        list(st.track_ids))
            for st in confirmed
        ]

    def drop_camera(self, camera_id: int) -> None:
        with self._lock:
            self._states.pop(camera_id, None)

    # ------------------------------------------------------------ внутреннее

    @staticmethod
    def _person_of(detections, st: _ObjState):
        """Человек, с которым связан объект (приоритет — распознанный)."""
        linked = [d for d in detections if d.track_id in st.track_ids]
        recognized = [d for d in linked if d.employee_id is not None]
        det = (recognized or linked or [None])[0]
        if det is None:
            return None
        return {
            "track_id": det.track_id,
            "employee_id": det.employee_id,
            "employee_name": det.employee_name,
            "global_id": det.global_id,
        }


def build_scene_objects(raw_dets, name_of, frame_w: int, frame_h: int,
                        forced_category: str | None = None):
    """Сырые детекции YOLO → SceneObject (bbox 0..1, без людей).

    name_of — callable(class_id) → имя класса (COCO или модель оружия).
    forced_category — категория для всех детекций этой модели (у кастомной
    оружейной модели имена вроде "pistol", которых нет в WEAPON_LABELS, —
    без принудительной категории алерт не сработал бы)."""
    out = []
    for d in raw_dets:
        if d.class_id == 0:
            continue  # люди обрабатываются трекером ByteTrack
        x1, y1, x2, y2 = d.bbox
        label = name_of(int(d.class_id))
        out.append(SceneObject(
            label=label,
            label_ru=ru_label(label),
            category=forced_category or category_of(label),
            confidence=round(float(d.score), 3),
            bbox=(round(max(0.0, x1 / frame_w), 4), round(max(0.0, y1 / frame_h), 4),
                  round(min(1.0, x2 / frame_w), 4), round(min(1.0, y2 / frame_h), 4)),
        ))
    return out
