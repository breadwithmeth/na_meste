"""Детекция людей: YOLOv8n ONNX через onnxruntime (без torch).

Вход модели: 1x3x640x640 float32 (0..1), выход: 1x84x8400
(4 координаты cx,cy,w,h + 80 коэффициентов классов COCO).
Детектор отдаёт только людей (класс 0).
"""
import logging
from typing import Optional

import cv2
import numpy as np
import onnxruntime as ort

logger = logging.getLogger("app.ai.detector")

PERSON_CLASS_ID = 0  # COCO


class PersonDetector:
    def __init__(
        self,
        model_path: str,
        providers: Optional[list[str]] = None,
        confidence: float = 0.35,
        input_size: int = 640,
    ):
        self.confidence = confidence
        self.input_size = input_size
        self.nms_threshold = 0.45
        self.session = ort.InferenceSession(
            model_path, providers=providers or ["CPUExecutionProvider"]
        )
        self.input_name = self.session.get_inputs()[0].name
        logger.info(
            "YOLO загружена: %s (провайдер: %s, порог: %.2f, класс: person)",
            model_path, self.session.get_providers()[0], confidence,
        )

    def detect(self, frame: np.ndarray) -> list[tuple[np.ndarray, float]]:
        """Кадр BGR → [(bbox_xyxy, score), ...] только люди."""
        blob, scale, (pad_x, pad_y) = self._letterbox(frame)
        outputs = self.session.run(None, {self.input_name: blob})
        pred = np.squeeze(outputs[0], 0).T  # (N, 4+C)

        class_scores = pred[:, 4:]
        class_ids = class_scores.argmax(axis=1)
        scores = class_scores.max(axis=1)
        mask = (class_ids == PERSON_CLASS_ID) & (scores >= self.confidence)
        if not mask.any():
            return []
        boxes = pred[mask, :4]
        scores = scores[mask]

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

        h, w = frame.shape[:2]
        result: list[tuple[np.ndarray, float]] = []
        keep = cv2.dnn.NMSBoxes(
            xywh.tolist(), scores.tolist(),
            self.confidence, self.nms_threshold,
        )
        for i in np.asarray(keep).flatten():
            x, y, bw, bh = xywh[int(i)]
            x1, y1 = max(0, int(x)), max(0, int(y))
            x2, y2 = min(w, int(x + bw)), min(h, int(y + bh))
            if x2 - x1 < 4 or y2 - y1 < 4:
                continue
            result.append((np.array([x1, y1, x2, y2]), float(scores[int(i)])))
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
