"""Person Re-Identification: OSNet (ONNX) — appearance-эмбеддинги людей.

Описывает одежду/фигуру человека, поэтому работает и когда лицо не видно —
в отличие от распознавания лиц. Модель osnet_x0_25_msmt17 (~1 МБ, 512-d).
У модели фиксированный batch — дополняем нулями и берём первые N строк.
"""
import logging
from typing import Optional

import cv2
import numpy as np
import onnxruntime as ort

logger = logging.getLogger("app.ai.reid")

INPUT_H, INPUT_W = 256, 128
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
EMBEDDING_DIM = 512


class PersonReID:
    def __init__(self, model_path: str, providers: Optional[list[str]] = None):
        self.session = ort.InferenceSession(
            model_path, providers=providers or ["CPUExecutionProvider"]
        )
        self.input_name = self.session.get_inputs()[0].name
        batch = self.session.get_inputs()[0].shape[0]
        self.batch = int(batch) if isinstance(batch, int) else 16
        logger.info(
            "Re-ID модель загружена: %s (batch=%d, провайдер=%s)",
            model_path, self.batch, self.session.get_providers()[0],
        )

    def embed(self, crops: list[np.ndarray]) -> np.ndarray:
        """Кропы людей (BGR, любые размеры) → (N, 512) L2-нормированные.

        Один batch-инференс на вызов — на кадре обычно 1–3 человека.
        """
        if not crops:
            return np.zeros((0, EMBEDDING_DIM), dtype=np.float32)
        n = min(len(crops), self.batch)
        batch = np.zeros((self.batch, 3, INPUT_H, INPUT_W), dtype=np.float32)
        for i in range(n):
            img = cv2.resize(crops[i], (INPUT_W, INPUT_H))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            batch[i] = ((img - _MEAN) / _STD).transpose(2, 0, 1)
        output = self.session.run(None, {self.input_name: batch})[0][:n]
        norms = np.linalg.norm(output, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return (output / norms).astype(np.float32)
