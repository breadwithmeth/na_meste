"""Детекция людей и предметов: YOLOv8n ONNX через onnxruntime (без torch).

Вход модели: 1x3x640x640 float32 (0..1), выход: 1x84x8400
(4 координаты cx,cy,w,h + 80 коэффициентов классов COCO).

По умолчанию детектор фильтрует только людей (класс 0) — поведение
прежних версий. При включённом списке классов (objects/оружие) та же
инференс-проходка отдаёт все включённые классы: лишнего инференса нет.
"""
import logging
from collections import defaultdict
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np
import onnxruntime as ort

logger = logging.getLogger("app.ai.detector")

PERSON_CLASS_ID = 0  # COCO


# 80 классов COCO в порядке выходных индексов модели
COCO_CLASSES = (
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep",
    "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
    "sports ball", "kite", "baseball bat", "baseball glove", "skateboard",
    "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork",
    "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv",
    "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
    "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
    "scissors", "teddy bear", "hair drier", "toothbrush",
)

COCO_CLASS_IDS = {name: idx for idx, name in enumerate(COCO_CLASSES)}


@dataclass
class RawDetection:
    """Сырая детекция: bbox xyxy в пикселях кадра + класс."""
    bbox: np.ndarray
    score: float
    class_id: int

    @property
    def class_name(self) -> str:
        return COCO_CLASSES[self.class_id] if 0 <= self.class_id < len(COCO_CLASSES) else str(self.class_id)


class PersonDetector:
    def __init__(
        self,
        model_path: str,
        providers: Optional[list[str]] = None,
        confidence: float = 0.35,
        input_size: int = 640,
        enabled_classes: Optional[set[int]] = None,
        class_names: Optional[list[str]] = None,
    ):
        """enabled_classes: None — только person для COCO-модели (как раньше);
        задан set id — все перечисленные классы. Для не-COCO модели (слот
        WEAPON_MODEL, class_names задан) None = все классы этой модели.
        class_names — имена классов по порядку выходных индексов."""
        self.confidence = confidence
        self.input_size = input_size
        self.nms_threshold = 0.45
        self.enabled_classes = enabled_classes
        self.class_names = class_names  # None → имена берутся из COCO_CLASSES
        self.session = ort.InferenceSession(
            model_path, providers=providers or ["CPUExecutionProvider"]
        )
        self.input_name = self.session.get_inputs()[0].name
        logger.info(
            "YOLO загружена: %s (провайдер: %s, порог: %.2f, классов: %s)",
            model_path, self.session.get_providers()[0], confidence,
            "person" if enabled_classes is None else len(enabled_classes),
        )

    def name_of(self, class_id: int) -> str:
        if self.class_names is not None:
            return (
                self.class_names[class_id]
                if 0 <= class_id < len(self.class_names) else str(class_id)
            )
        return COCO_CLASSES[class_id] if 0 <= class_id < len(COCO_CLASSES) else str(class_id)

    def detect(self, frame: np.ndarray) -> list[tuple[np.ndarray, float]]:
        """Кадр BGR → [(bbox_xyxy, score), ...] только люди.

        Совместимо с прежним поведением (RecognitionService, тесты)."""
        return [
            (d.bbox, d.score)
            for d in self.detect_all(frame)
            if d.class_id == PERSON_CLASS_ID
        ]

    def detect_all(self, frame: np.ndarray) -> list[RawDetection]:
        """Кадр BGR → детекции всех включённых классов (per-class NMS)."""
        blob, scale, (pad_x, pad_y) = self._letterbox(frame)
        outputs = self.session.run(None, {self.input_name: blob})
        pred = np.squeeze(outputs[0], 0).T  # (N, 4+C)

        class_scores = pred[:, 4:]
        class_ids = class_scores.argmax(axis=1)
        scores = class_scores.max(axis=1)
        if self.enabled_classes is None and self.class_names is None:
            mask = (class_ids == PERSON_CLASS_ID) & (scores >= self.confidence)
        else:
            mask = scores >= self.confidence
            if self.enabled_classes is not None:
                mask &= np.isin(class_ids, list(self.enabled_classes))
        if not mask.any():
            return []
        boxes = pred[mask, :4]
        scores = scores[mask]
        det_class_ids = class_ids[mask]

        # cx,cy,w,h (координаты letterbox) → xyxy кадра: отступ вычитается
        # только из координат, размеры делятся на scale как есть
        # (вычитание pad из w/h ломало боксы на 16:9-потоках: объекты
        # ниже pad/scale получали отрицательную высоту и отбрасывались)
        xywh = np.stack(
            [boxes[:, 0] - boxes[:, 2] / 2, boxes[:, 1] - boxes[:, 3] / 2,
             boxes[:, 2], boxes[:, 3]], axis=1,
        )
        xywh[:, [0, 1]] = (xywh[:, [0, 1]] - np.array([pad_x, pad_y])) / scale
        xywh[:, [2, 3]] /= scale

        # per-class NMS: без этого бокс человека подавил бы нож/чашку в руке
        by_class: dict[int, list[int]] = defaultdict(list)
        for i, cid in enumerate(det_class_ids):
            by_class[int(cid)].append(i)

        h, w = frame.shape[:2]
        result: list[RawDetection] = []
        for cid, idxs in by_class.items():
            keep = cv2.dnn.NMSBoxes(
                xywh[idxs].tolist(), scores[idxs].tolist(),
                self.confidence, self.nms_threshold,
            )
            for i in np.asarray(keep).flatten():
                j = idxs[int(i)]
                x, y, bw, bh = xywh[j]
                x1, y1 = max(0, int(x)), max(0, int(y))
                x2, y2 = min(w, int(x + bw)), min(h, int(y + bh))
                if x2 - x1 < 4 or y2 - y1 < 4:
                    continue
                result.append(RawDetection(
                    bbox=np.array([x1, y1, x2, y2]),
                    score=float(scores[j]),
                    class_id=cid,
                ))
        return result

    def _letterbox(self, frame: np.ndarray):
        size = self.input_size
        h, w = frame.shape[:2]
        scale = min(size / w, size / h)
        nw, nh = int(round(w * scale)), int(round(h * scale))
        img = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_LINEAR)
        pad_x, pad_y = (size - nw) // 2, (size - nh) // 2
        canvas = np.full((size, size, 3), 114, dtype=np.uint8)
        canvas[pad_y:pad_y + nh, pad_x:pad_x + nw] = img
        blob = canvas.transpose(2, 0, 1)[None].astype(np.float32) / 255.0
        return blob, scale, (pad_x, pad_y)
