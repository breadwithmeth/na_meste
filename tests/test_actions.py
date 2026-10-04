"""Тесты распознавания действий: классификатор позы + сглаживание + БД + API.

Запуск (без pytest, только зависимости проекта):

    venv/Scripts/python tests/test_actions.py      # Windows
    venv/bin/python tests/test_actions.py          # macOS / Linux

Проверяется:
- classify_pose: стоит/идёт/сидит/лежит/ест/телефон по синтетическим скелетам;
- зона рук (desk/down) для производных действий working/resting;
- ActionRecognizer: голосование, гистерезис, working/resting/eating во времени,
  запись наблюдений (смена действия + троттлинг), drop_camera;
- ActionLogManager: запись в БД и очистка старых наблюдений;
- API: история с фильтрами, live из DetectionState, сводка за период.
"""
import sys
import traceback
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.ai.actions import (
    ActionRecognizer, classify_pose, ACTION_LABELS,
)
from app.api import actions as api
from app.database.database import Base
from app.database import models as db_models  # noqa: F401 — регистрация таблиц
from app.database.models import ActionObservation, Camera, Employee
from app.ai.recognition_service import DetectionResult
from app.services.action_service import ActionLogManager
from app.services.presence_service import utcnow

# ------------------------------------------------------------------ стенды

FRAME = np.zeros((720, 1280, 3), dtype=np.uint8)
W, H = 1280, 720

# индексы COCO
KPTS_NAMES = dict(nose=0, l_ear=3, r_ear=4, l_sh=5, r_sh=6, l_el=7, r_el=8,
                  l_wr=9, r_wr=10, l_hip=11, r_hip=12, l_kn=13, r_kn=14,
                  l_an=15, r_an=16)


def skeleton(points: dict, conf: float = 0.9) -> np.ndarray:
    """Скелет (17, 3): точки по имени, пропущенные — conf=0 (не видны)."""
    kpts = np.zeros((17, 3), dtype=np.float32)
    kpts[:, 2] = 0.0
    for name, (x, y) in points.items():
        i = KPTS_NAMES[name]
        kpts[i] = (x, y, conf)
    return kpts


def standing(x_shift: float = 0.0, wrist_y: float = 450.0) -> np.ndarray:
    """Стоящий человек по центру кадра (bbox ≈ (400,100)-(560,620))."""
    s = x_shift
    return skeleton(dict(
        nose=(480 + s, 140), l_ear=(455 + s, 135), r_ear=(505 + s, 135),
        l_sh=(440 + s, 240), r_sh=(520 + s, 240),
        l_el=(420 + s, 330), r_el=(540 + s, 330),
        l_wr=(430 + s, wrist_y), r_wr=(530 + s, wrist_y),
        l_hip=(445 + s, 420), r_hip=(515 + s, 420),
        l_kn=(450 + s, 590), r_kn=(510 + s, 590),
        l_an=(450 + s, 610), r_an=(510 + s, 610),
    ))


def sitting(wrists: tuple = "desk") -> np.ndarray:
    """Сидящий: бёдра низко, колени у бёдер, широкий бокс."""
    if wrists == "desk":
        lw, rw = (460, 470), (540, 470)
    elif wrists == "down":
        lw, rw = (430, 560), (570, 560)
    else:  # явные координаты
        lw, rw = wrists
    return skeleton(dict(
        nose=(500, 340), l_ear=(475, 335), r_ear=(525, 335),
        l_sh=(440, 420), r_sh=(560, 420),
        l_el=(430, 470), r_el=(570, 470),
        l_wr=lw, r_wr=rw,
        l_hip=(450, 530), r_hip=(550, 530),
        l_kn=(450, 545), r_kn=(550, 545),
        l_an=(430, 560), r_an=(570, 560),
    ))


def lying() -> np.ndarray:
    """Лежащий: торс горизонтален."""
    return skeleton(dict(
        nose=(240, 390), l_ear=(265, 385), r_ear=(295, 385),
        l_sh=(320, 400), r_sh=(420, 400),
        l_el=(460, 430), r_el=(520, 440),
        l_wr=(570, 450), r_wr=(590, 470),
        l_hip=(600, 410), r_hip=(700, 410),
        l_kn=(740, 460), r_kn=(780, 470),
        l_an=(800, 500), r_an=(830, 510),
    ))


STAND_BBOX = (400, 100, 560, 620)
SIT_BBOX = (380, 300, 620, 560)
LYING_BBOX = (220, 300, 860, 560)


class FakePose:
    """Поза по расписанию: estimate() возвращает заранее заданные скелеты."""

    def __init__(self, frames: list):
        self.frames = list(frames)
        self._i = 0

    def estimate(self, crop):
        kpts = self.frames[min(self._i, len(self.frames) - 1)]
        self._i += 1
        return kpts.copy()


def make_det(track_id: int = 1, bbox=STAND_BBOX, employee_id=None,
             global_id=None) -> DetectionResult:
    """DetectionResult с пиксельным bbox, переведённым в нормализованный."""
    x1, y1, x2, y2 = bbox
    return DetectionResult(
        track_id=track_id, employee_id=employee_id, employee_name=None,
        confidence=None,
        bbox=(x1 / W, y1 / H, x2 / W, y2 / H),
        state="unknown", global_id=global_id,
    )


def run_frames(recognizer, pose_frames, *, n=None, bbox=STAND_BBOX, track_id=1,
               employee_id=None, global_id=None):
    """Прогнать распознаватель по секундам; возвращает (последний det, записи)."""
    frames = pose_frames if n is None else pose_frames[:n]
    pose = FakePose(frames)
    recognizer.pose = pose
    records = []
    recognizer.on_action = lambda *args: records.append(args)
    det = None
    now = 1000.0  # произвольная точка отсчёта monotonic
    for i in range(len(frames)):
        det = make_det(track_id, bbox, employee_id, global_id)
        recognizer.process(1, [det], FRAME, now + i)
    return det, records


def make_recognizer(**kwargs) -> ActionRecognizer:
    params = dict(
        interval=1.0, smooth_seconds=6.0, rest_after=15.0, log_interval=10.0,
    )
    params.update(kwargs)
    return ActionRecognizer(FakePose([]), lambda *a: None, **params)


# ------------------------------------------------------------------- тесты

def test_classify_instant():
    # стоит: вертикальный торс, бёдра высоко, узкий бокс
    action, _, hands = classify_pose(standing(), STAND_BBOX)
    assert action == "standing", f"ожидался standing, получен {action}"
    assert hands == "down", "руки опущены, а не {hands}".format(hands=hands)

    # стоит с руками на прилавке (между плечами и бёдрами) — зона desk
    action, _, hands = classify_pose(standing(wrist_y=380), STAND_BBOX)
    assert action == "standing" and hands == "desk", (action, hands)

    # идёт: та же поза, но заметное перемещение
    action, _, _ = classify_pose(standing(), STAND_BBOX, speed=1.5)
    assert action == "walking", f"ожидался walking, получен {action}"

    # сидит: бёдра низко (hip_ratio 0.88), колени у бёдер, бокс квадратный
    action, conf, hands = classify_pose(sitting(), SIT_BBOX)
    assert action == "sitting", f"ожидался sitting, получен {action}"
    assert hands == "desk", f"руки на столе → desk, получен {hands}"
    assert conf >= 0.45

    # лежит: торс горизонтален
    action, _, _ = classify_pose(lying(), LYING_BBOX)
    assert action == "lying", f"ожидался lying, получен {action}"

    # ест: кисть у рта (ниже носа, у центра лица)
    eating = standing(wrist_y=170)
    action, _, hands = classify_pose(eating, STAND_BBOX)
    assert action == "eating", f"ожидался eating, получен {action}"
    assert hands == "face"

    # телефон: кисть на уровне уха сбоку
    phoning = standing(wrist_y=130)
    # сместим правую кисть к правому уху (505, 135)
    phoning[KPTS_NAMES["r_wr"]] = (515, 130, 0.9)
    action, _, _ = classify_pose(phoning, STAND_BBOX)
    assert action == "phone", f"ожидался phone, получен {action}"


def test_recognizer_working():
    """Сидит с руками в рабочей зоне → работает; запись в БД на смене."""
    rec = make_recognizer()
    det, records = run_frames(rec, [sitting()] * 5)
    assert det.action == "working", f"ожидался working, получен {det.action}"
    assert det.action_confidence is not None and det.action_since is not None
    # первая запись — при появлении действия
    assert any(r[4] == "working" for r in records), records
    # троттлинг: за 5 сек при неизменном действии — одна запись
    assert len(records) == 1, f"ожидалась 1 запись, получено {len(records)}"


def test_recognizer_resting():
    """Сидит без активности рук дольше rest_after → отдыхает."""
    rec = make_recognizer(rest_after=15.0)
    det, records = run_frames(rec, [sitting(wrists="down")] * 17)
    assert det.action == "resting", f"ожидался resting, получен {det.action}"
    # сидит (t=0) → отдыхает (t=15): обе записи присутствуют
    actions = [r[4] for r in records]
    assert "sitting" in actions and "resting" in actions, actions


def test_recognizer_eating_intermittent():
    """Рука к лицу в трети кадров (между укусами) → ест/пьёт."""
    frames = [sitting(wrists=((470, 380), (530, 470))) for _ in range(6)]
    # каждый второй кадр — кисть у рта (нос на (500, 340))
    for i in range(0, 6, 2):
        frames[i] = sitting(wrists=((470, 370), (530, 470)))
    rec = make_recognizer()
    det, _ = run_frames(rec, frames)
    assert det.action == "eating", f"ожидался eating, получен {det.action}"


def test_recognizer_walking_and_hysteresis():
    # идёт: центр тела смещается на 150 px/сек (торс 180 px → 0.83 торса/сек)
    frames = [standing(x_shift=i * 150.0) for i in range(5)]
    rec = make_recognizer()
    det, _ = run_frames(rec, frames)
    assert det.action == "walking", f"ожидался walking, получен {det.action}"

    # гистерезис: 5 голосов «сидит» → 4 голоса «стоит» — переключение только
    # когда «стоит» строго перевесит в окне 6 сек
    frames = [sitting()] * 5 + [standing()] * 4
    rec = make_recognizer()
    pose = FakePose(frames)
    rec.pose = pose
    det = None
    now = 2000.0
    for i in range(9):
        det = make_det()
        rec.process(1, [det], FRAME, now + i)
        if i == 6:  # окно: сидит×3 (t=2..4) против стоит×3 (t=5..6) — ничья
            assert det.action in ("sitting", "working"), det.action
    assert det.action == "standing", f"ожидался standing, получен {det.action}"


def test_recognizer_drop_camera_and_ids():
    rec = make_recognizer(log_interval=10.0)
    # сотрудник опознан на 3-й секунде → повторная запись из-за смены ids
    det, records = run_frames(rec, [sitting()] * 3, employee_id=7)
    assert records and records[0][2] == 7, records
    rec.drop_camera(1)
    assert not rec._states, "состояния треков камеры должны быть удалены"


def test_log_manager_and_cleanup():
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine, expire_on_commit=False)
    with factory() as db:
        db.add(Camera(id=1, name="Касса", nvr_host="h", username="u", password="p",
                      channel=1))
        db.add(Employee(id=7, name="Иван Петров"))
        db.commit()

    manager = ActionLogManager(factory, keep_days=30)
    manager.record(1, 17, 7, None, "working", 0.8)
    manager.record(1, 17, 7, None, "sitting", 0.7)
    with factory() as db:
        assert db.query(ActionObservation).count() == 2
        # старое наблюдение — удаляется очисткой
        old = ActionObservation(camera_id=1, track_id=3, action="lying",
                                created_at=utcnow() - timedelta(days=40))
        db.add(old)
        db.commit()
    manager.cleanup()
    with factory() as db:
        assert db.query(ActionObservation).count() == 2

    # ---- API: история + фильтры + сводка ----
    with factory() as db:
        rows = api.api_actions_history(db=db)
        assert len(rows) == 2
        assert rows[0]["employee_name"] == "Иван Петров"
        assert rows[0]["camera_name"] == "Касса"
        assert rows[0]["action_label"] == ACTION_LABELS[rows[0]["action"]]

        rows = api.api_actions_history(action="working", db=db)
        assert len(rows) == 1 and rows[0]["action"] == "working"

        rows = api.api_actions_history(employee_id=999, db=db)
        assert rows == []

        summary = api.api_actions_summary(hours=8, db=db)
        assert len(summary) == 1
        person = summary[0]
        assert person["employee_name"] == "Иван Петров"
        assert person["total_seconds"] == 20  # 2 наблюдения × ACTION_LOG_INTERVAL
        assert list(person["actions"]) == ["working", "sitting"]

    # ---- API: live из DetectionState ----
    class FakeState:
        def snapshot(self):
            return [{
                "camera_id": 1, "tracks": [
                    {"track_id": 17, "employee_id": 7, "employee_name": "Иван",
                     "global_id": None, "action": "working",
                     "action_confidence": 0.8, "action_since": 1700000000.0},
                    {"track_id": 18, "employee_id": None, "employee_name": None,
                     "global_id": 4, "action": None,
                     "action_confidence": None, "action_since": None},
                ],
            }]

    request = SimpleNamespace(app=SimpleNamespace(
        state=SimpleNamespace(detection_state=FakeState())))
    with factory() as db:
        live = api.api_actions_live(request=request, db=db)
    assert len(live) == 1, live
    assert live[0]["action_label"] == "работает"
    assert live[0]["camera_name"] == "Касса"
    assert live[0]["action_since"] is not None

    request_none = SimpleNamespace(
        app=SimpleNamespace(state=SimpleNamespace(detection_state=None)))
    with factory() as db:
        assert api.api_actions_live(request=request_none, db=db) == []


# --------------------------------------------------------------------- main

def main() -> int:
    tests = [
        test_classify_instant,
        test_recognizer_working,
        test_recognizer_resting,
        test_recognizer_eating_intermittent,
        test_recognizer_walking_and_hysteresis,
        test_recognizer_drop_camera_and_ids,
        test_log_manager_and_cleanup,
    ]
    failed = 0
    for test in tests:
        try:
            test()
            print(f"PASS  {test.__name__}")
        except Exception:
            failed += 1
            print(f"FAIL  {test.__name__}")
            traceback.print_exc()
    print(f"\n{len(tests) - failed}/{len(tests)} тестов пройдено")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
