"""Компактная реализация ByteTrack: IoU-трекинг с двухступенчатой ассоциацией.

Особенности под задачу присутствия:
- lost-трек живёт track_lost_timeout секунд (grace period) — если человек
  пропал ненадолго, это тот же track, и employee_id на нём сохраняется;
- счётчик hits отсеивает шумовые детекции;
- привязка сотрудника живёт на треке, пока трек существует.
"""
import logging
import time
from typing import Optional

import numpy as np

logger = logging.getLogger("app.ai.tracker")


def iou_xyxy(a, b) -> float:
    """IoU двух боксов в формате xyxy."""
    x1 = max(a[0], b[0]); y1 = max(a[1], b[1])
    x2 = min(a[2], b[2]); y2 = min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if inter <= 0:
        return 0.0
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    return inter / (area_a + area_b - inter)


class Track:
    __slots__ = (
        "track_id", "bbox", "score", "state", "last_seen", "hits", "is_new",
        "employee_id", "employee_conf", "candidate_id", "candidate_count",
        "last_recog_attempt", "best_unknown_sim", "first_seen",
        "unknown_reported", "unknown_fails",
    )

    def __init__(self, track_id: int, bbox: np.ndarray, score: float, now: float):
        self.track_id = track_id
        self.bbox = bbox
        self.score = score
        self.state = "tracked"           # tracked | lost
        self.last_seen = now
        self.hits = 1
        self.is_new = True
        self.first_seen = now
        # результат распознавания (живёт на треке — пока трек существует,
        # повторно распознавать не нужно)
        self.employee_id: Optional[int] = None
        self.employee_conf: Optional[float] = None
        self.candidate_id: Optional[int] = None
        self.candidate_count = 0
        self.last_recog_attempt: float = 0.0
        self.best_unknown_sim: Optional[float] = None
        # фиксация постороннего: одно событие на трек
        self.unknown_reported = False
        self.unknown_fails = 0


class ByteTracker:
    """Трекер одной камеры. update() вызывается на каждый обработанный AI-кадр."""

    def __init__(
        self,
        lost_timeout: float = 3.0,
        high_threshold: float = 0.5,
        low_threshold: float = 0.1,
        match_threshold: float = 0.3,
    ):
        self.lost_timeout = lost_timeout
        self.high_threshold = high_threshold
        self.low_threshold = low_threshold
        self.match_threshold = match_threshold
        self.tracks: list[Track] = []
        self._next_id = 1

    def update(
        self, detections: list[tuple[np.ndarray, float]], now: Optional[float] = None
    ) -> list[Track]:
        """detections: [(bbox_xyxy, score)]. Возвращает треки в состоянии tracked."""
        now = time.monotonic() if now is None else now
        for t in self.tracks:
            t.is_new = False

        high_idx = [i for i, (_b, s) in enumerate(detections) if s >= self.high_threshold]
        low_idx = [
            i for i, (_b, s) in enumerate(detections)
            if self.low_threshold <= s < self.high_threshold
        ]

        tracked = [t for t in self.tracks if t.state == "tracked"]
        lost = [t for t in self.tracks if t.state == "lost"]

        # стадия 1: уверенные детекции × текущие треки
        matches_1, rest_high = self._match(
            tracked, high_idx, detections, now
        )
        # стадия 2: слабые детекции × (непроклассифицированные + потерянные треки)
        rest_tracked = [t for t in tracked if t not in matches_1]
        matches_2, _rest_low = self._match(
            rest_tracked + lost, low_idx, detections, now
        )

        # треки без детекции → lost
        matched_now = matches_1 + matches_2
        for t in tracked:
            if t not in matched_now:
                t.state = "lost"

        # новые треки из непроклассифицированных уверенных детекций
        for j in rest_high:
            bbox, score = detections[j]
            self.tracks.append(Track(self._next_id, bbox, score, now))
            self._next_id += 1

        # убрать lost-треки с истёкшим grace period
        self.tracks = [
            t for t in self.tracks
            if t.state == "tracked" or (now - t.last_seen) <= self.lost_timeout
        ]
        return [t for t in self.tracks if t.state == "tracked"]

    # ------------------------------------------------------------ внутреннее

    def _match(
        self,
        tracks: list[Track],
        det_indices: list[int],
        detections: list[tuple[np.ndarray, float]],
        now: float,
    ) -> tuple[list[Track], list[int]]:
        """Жадное IoU-сопоставление треков с детекциями.

        Возвращает (совпавшие треки, индексы непроклассифицированных детекций).
        """
        if not tracks or not det_indices:
            return [], list(det_indices)

        pairs: list[tuple[float, Track, int]] = []
        for t in tracks:
            for j in det_indices:
                v = iou_xyxy(t.bbox, detections[j][0])
                if v >= self.match_threshold:
                    pairs.append((v, t, j))
        pairs.sort(key=lambda p: -p[0])

        matched: list[Track] = []
        used_tracks: set[int] = set()
        used_dets: set[int] = set()
        for v, t, j in pairs:
            if id(t) in used_tracks or j in used_dets:
                continue
            bbox, score = detections[j]
            t.bbox = bbox
            t.score = score
            t.last_seen = now
            t.hits += 1
            t.state = "tracked"
            used_tracks.add(id(t))
            used_dets.add(j)
            matched.append(t)

        return matched, [j for j in det_indices if j not in used_dets]
