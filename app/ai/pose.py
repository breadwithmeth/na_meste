"""Оценка позы человека: YOLOv8n-pose ONNX (17 ключевых точек COCO).

Вход модели: 1x3x640x640 float32 (0..1), выход: 1x56x8400
(4 координаты cx,cy,w,h + 1 score + 17×3 keypoints x,y,conf).

Запускается на кропе человека из кадра — люди уже найдены YOLO и
сопровождаются трекером, поэтому из всех людей в кропе берём лучший
по score. Ключевые точки возвращаются в координатах кропа.
"""
import logging
from typing import Optional

import cv2
import numpy as np
import onnxruntime as ort

logger = logging.getLogger("app.ai.pose")

NUM_KEYPOINTS = 17            # COCO: nose, eyes, ears, shoulders, elbows,
                              # wrists, hips, knees, ankles


class PoseEstimator:
    def __init__(
        self,
        model_path: str,
        providers: Optional[list[str]] = None,
        input_size: int = 640,
    ):
        self.input_size = input_size
        self.session = ort.InferenceSession(
            model_path, providers=providers or ["CPUExecutionProvider"]
        )
        self.input_name = self.session.get_inputs()[0].name
        # 5 + 17*3 = 56 (box + score + keypoints) — проверяем формат выхода
        num_pred = self.session.get_outputs()[0].shape[1]
        if not isinstance(num_pred, int) or num_pred < 5 + NUM_KEYPOINTS * 3:
            raise ValueError(f"Неожиданный формат выхода pose-модели: {num_pred}")
        logger.info(
            "Pose-модель загружена: %s (провайдер: %s)",
            model_path, self.session.get_providers()[0],
        )

    def estimate(self, crop: np.ndarray) -> Optional[np.ndarray]:
        """Кроп человека (BGR) → keypoints (17, 3) [x, y, conf] в координатах
        кропа. None — человек в кропе не найден."""
        if crop is None or crop.size == 0:
            return None
        blob, scale, (pad_x, pad_y) = self._letterbox(crop)
        outputs = self.session.run(None, {self.input_name: blob})
        pred = np.squeeze(outputs[0], 0).T  # (N, 56)

        best = int(np.argmax(pred[:, 4]))
        if float(pred[best, 4]) < 0.25:
            return None  # в кропе нет уверенного человека

        kpts = pred[best, 5:5 + NUM_KEYPOINTS * 3].reshape(NUM_KEYPOINTS, 3)
        # координаты letterbox → координаты кропа
        kpts[:, 0] = (kpts[:, 0] - pad_x) / scale
        kpts[:, 1] = (kpts[:, 1] - pad_y) / scale
        return kpts.astype(np.float32)

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
