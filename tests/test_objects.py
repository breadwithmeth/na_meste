"""Тесты предметов сцены и атрибутов внешности.

Запуск (без pytest, только зависимости проекта):

    venv/Scripts/python tests/test_objects.py      # Windows
    venv/bin/python tests/test_objects.py          # macOS / Linux

Проверяется:
- категории/русские подписи предметов, нейминг цветов по синтетическим патчам;
- build_scene_objects: фильтр людей, нормализация bbox, имена классов;
- SceneObjectTracker: антифликер (MIN_HITS), привязка «при ком», событие
  оружия с человеком, TTL исчезновения, drop_camera;
- RecognitionService: одна YOLO-проходка → люди + предметы (фейковый
  детектор), пол/возраст (голосование + EMA);
- AttributeExtractor: стабилизация цвета, запись в БД;
- ObjectEventManager / AppearanceLogManager: БД + очистка + API.
"""
import sys
import time
import traceback
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.ai.attributes import AttributeExtractor, name_color
from app.ai.objects import (
    SceneObject, SceneObjectTracker, build_scene_objects, category_of, ru_label,
)
from app.ai.recognition_service import DetectionResult, RecognitionService
from app.ai.detector import RawDetection
from app.api import appearance as appearance_api
from app.api import objects as objects_api
from app.database.database import Base
from app.database import models as db_models  # noqa: F401 — регистрация таблиц
from app.database.models import AppearanceObservation, Camera, Employee, ObjectObservation
from app.services.appearance_service import AppearanceLogManager
from app.services.object_service import ObjectEventManager
from app.services.presence_service import utcnow

FRAME = np.zeros((720, 1280, 3), dtype=np.uint8)
W, H = 1280, 720

PERSON_BBOX = np.array([500, 100, 700, 600], dtype=float)   # человек в кадре
KNIFE_BBOX = np.array([560, 400, 620, 480], dtype=float)    # нож в руке


def det(track_id=1, employee_id=None, bbox=(0.39, 0.14, 0.55, 0.83)):
    return DetectionResult(
        track_id=track_id, employee_id=employee_id,
        employee_name="Иван" if employee_id else None,
        confidence=None, bbox=bbox, state="unknown" if employee_id is None else "detecting",
    )


def scene_raw(class_id, bbox, score=0.8):
    return RawDetection(bbox=np.asarray(bbox, dtype=float), score=score, class_id=class_id)


def name_of(class_id):
    return {0: "person", 43: "knife", 26: "backpack"}.get(class_id, f"c{class_id}")


# ------------------------------------------------------------------ базовые

def test_categories_and_colors():
    assert category_of("knife") == "weapon"
    assert category_of("scissors") == "weapon"
    assert category_of("banana") == "food"
    assert category_of("cup") == "tableware"
    assert category_of("backpack") == "carried"
    assert category_of("dining table") == "furniture"
    assert ru_label("knife") == "нож"
    assert ru_label("whatever_new") == "whatever_new"

    blue = np.zeros((20, 20, 3), dtype=np.uint8)
    blue[:, :] = (255, 80, 30)          # BGR: синий
    assert name_color(blue)[0] == "синий"
    red = np.zeros((20, 20, 3), dtype=np.uint8)
    red[:, :] = (60, 60, 220)           # BGR: красный
    assert name_color(red)[0] == "красный"
    black = np.full((20, 20, 3), 15, dtype=np.uint8)
    assert name_color(black)[0] == "чёрный"
    white = np.full((20, 20, 3), 240, dtype=np.uint8)
    assert name_color(white)[0] == "белый"
    gray = np.full((20, 20, 3), 120, dtype=np.uint8)
    assert name_color(gray)[0] == "серый"
    brown = np.zeros((20, 20, 3), dtype=np.uint8)
    brown[:, :] = (30, 70, 110)         # тёмный красно-оранжевый
    assert name_color(brown)[0] == "коричневый"
    hexv = name_color(blue)[1]
    assert hexv.startswith("#") and len(hexv) == 7
    assert name_color(np.zeros((0,), dtype=np.uint8)) is None


def test_build_scene_objects():
    raws = [scene_raw(0, PERSON_BBOX), scene_raw(43, KNIFE_BBOX, 0.7)]
    objs = build_scene_objects(raws, name_of, W, H)
    assert len(objs) == 1, "человек не должен попадать в предметы"
    obj = objs[0]
    assert obj.label == "knife" and obj.label_ru == "нож"
    assert obj.category == "weapon" and obj.confidence == 0.7
    # нормализация: 560/1280, 400/720, 620/1280, 480/720
    assert obj.bbox == (round(560 / W, 4), round(400 / H, 4),
                        round(620 / W, 4), round(480 / H, 4))


def test_tracker_min_hits_and_items():
    events = []
    tracker = SceneObjectTracker(on_object=lambda *a: events.append(a))
    raws = [SceneObject("knife", "нож", "weapon", 0.8,
                        (560 / W, 400 / H, 620 / W, 480 / H))]
    detections = [det(track_id=1, employee_id=None)]

    out1 = tracker.process(1, raws, detections, FRAME, now=time.monotonic())
    assert out1 == [], "первое появление не подтверждено (антифликер)"
    assert events == []

    out2 = tracker.process(1, raws, detections, FRAME, now=time.monotonic())
    assert len(out2) == 1 and out2[0].label == "knife" and out2[0].label_ru == "нож"
    assert detections[0].items == ["нож"], "нож у человека → items"
    assert out2[0].track_ids == [1]
    assert len(events) == 1, "событие оружия — один раз при подтверждении"
    cam_id, obj, frame, person = events[0]
    assert cam_id == 1 and obj.label == "knife"
    assert person == {"track_id": 1, "employee_id": None,
                      "employee_name": None, "global_id": None}

    # объект «доживает» и дальше, но событие не повторяется
    out3 = tracker.process(1, raws, detections, FRAME, now=time.monotonic())
    assert len(out3) == 1 and len(events) == 1


def test_tracker_ttl_and_drop_camera():
    tracker = SceneObjectTracker()
    raws = [SceneObject("cup", "чашка", "tableware", 0.7,
                        (0.1, 0.1, 0.2, 0.2))]
    t0 = time.monotonic()
    tracker.process(1, raws, [], FRAME, now=t0)
    tracker.process(1, raws, [], FRAME, now=t0 + 0.1)
    assert len(tracker.process(1, raws, [], FRAME, now=t0 + 0.2)) == 1
    # TTL истёк (объект пропал из кадра) → пусто
    out = tracker.process(1, [], [], FRAME, now=t0 + 3.0)
    assert out == []
    tracker.process(2, raws, [], FRAME, now=t0 + 3.1)
    tracker.process(2, raws, [], FRAME, now=t0 + 3.2)
    tracker.drop_camera(2)
    assert 2 not in tracker._states, "состояния камеры должны быть удалены"


def test_recognition_service_split():
    """Одна инференс-проходка: люди трекаются, предметы уходят в outcome."""
    class FakeDetector:
        def __init__(self):
            self.calls = 0

        def name_of(self, class_id):
            return name_of(class_id)

        def detect_all(self, frame):
            self.calls += 1
            return [scene_raw(0, PERSON_BBOX, 0.9), scene_raw(43, KNIFE_BBOX, 0.7)]

        def detect(self, frame):
            return [(d.bbox, d.score) for d in self.detect_all(frame)
                    if d.class_id == 0]

    service = RecognitionService(
        FakeDetector(), SimpleNamespace(detect=lambda *a, **k: []),
        SimpleNamespace(employee_name=lambda eid: None),
        object_class_ids=[43],
    )
    t0 = time.monotonic()
    o1 = service.process_frame(FRAME, 1, now=t0)
    o2 = service.process_frame(FRAME, 1, now=t0 + 0.25)
    assert service.detector.calls == 2, "по одной инференс-проходке на кадр"
    assert len(o2.detections) == 1, "человек подтвердился на 2-м кадре (min_hits)"
    assert o2.objects and o2.objects[0].label == "knife" and o2.objects[0].label_ru == "нож"
    assert o2.objects[0].bbox[0] == round(560 / W, 4)

    # выключенные предметы: прежнее поведение, objects пуст
    service_off = RecognitionService(
        FakeDetector(), SimpleNamespace(detect=lambda *a, **k: []),
        SimpleNamespace(employee_name=lambda eid: None),
    )
    o = service_off.process_frame(FRAME, 1, now=t0)
    assert o.objects == []


def test_gender_age_collection():
    service = RecognitionService(
        SimpleNamespace(detect=lambda *a, **k: []),
        SimpleNamespace(detect=lambda *a, **k: []),
        SimpleNamespace(employee_name=lambda eid: None),
    )
    track = SimpleNamespace(gender=None, gender_votes=[], age=None)
    service._collect_attributes(track, SimpleNamespace(gender=1, age=30.0))
    service._collect_attributes(track, SimpleNamespace(gender=1, age=32.0))
    assert track.gender == 1
    assert abs(track.age - 30.6) < 0.01, "EMA возраста: 0.7*30 + 0.3*32"
    service._collect_attributes(track, SimpleNamespace(gender=0, age=32.0))
    assert track.gender == 1, "2 из 3 голосов — мужчина"
    assert abs(track.age - 31.02) < 0.01
    service._collect_attributes(track, SimpleNamespace(gender=0, age=None))
    assert track.gender == 0 and abs(track.age - 31.02) < 0.01, "age=None не меняет"

    result = service._to_result(SimpleNamespace(
        bbox=PERSON_BBOX, employee_id=None, employee_conf=None,
        last_recog_attempt=1, best_unknown_sim=0.4, track_id=5,
        gender=0, age=31.6), W, H)
    assert result.gender == "женщина" and result.age == 32


def test_attribute_extractor():
    appearance = []
    extractor = AttributeExtractor(
        on_attributes=lambda *a: appearance.append(a),
        interval=0.5, log_interval=30.0,
    )
    # человек крупным планом: торс синий, ноги чёрные (красим полосы целиком)
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    frame[140:285, 535:680] = (255, 80, 30)   # синий торс (вся полоса 0.15–0.45)
    d = det(track_id=1, employee_id=7, bbox=(0.42, 0.1, 0.53, 0.75))
    t0 = time.monotonic()
    for i in range(4):  # 4 сэмпла: стабилизация после 3
        extractor.process(1, [d], frame, now=t0 + i * 0.6)
    assert d.upper_color == "синий" and d.upper_hex
    assert d.lower_color == "чёрный"
    assert len(appearance) >= 1
    cam_id, track_id, emp_id, gid, upper, lower, gender, age = appearance[0]
    assert (cam_id, track_id, emp_id) == (1, 1, 7) and upper == "синий"

    extractor.drop_camera(1)
    assert extractor._states == {}


# ------------------------------------------------------------ БД + сервисы

def make_db():
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
    return factory


def test_object_manager_and_api():
    factory = make_db()
    sent = []
    manager = ObjectEventManager(
        factory, SimpleNamespace(send_photo=lambda jpeg, caption: sent.append((jpeg, caption))),
        keep_days=30, weapon_alert_enabled=True, weapon_alert_cooldown=60.0,
        event_cooldown=30.0,
    )
    obj = SceneObject("knife", "нож", "weapon", 0.8,
                      (560 / W, 400 / H, 620 / W, 480 / H), [1])
    person = {"track_id": 1, "employee_id": 7, "employee_name": "Иван Петров",
              "global_id": 3}
    frame = np.full((720, 1280, 3), 90, dtype=np.uint8)

    manager.record(1, obj, frame, person)
    manager.record(1, obj, frame, person)  # кулдаун — второй раз не пишется
    with factory() as db:
        assert db.query(ObjectObservation).count() == 1
        obs = db.query(ObjectObservation).first()
        assert obs.label == "knife" and obs.category == "weapon"
        assert obs.snapshot is not None, "для оружия сохраняется снимок"
        assert obs.employee_id == 7 and obs.global_id == 3
    assert len(sent) == 1 and sent[0][1].startswith("🚨")

    # еда: пишется без снимка, без Telegram
    food = SceneObject("pizza", "пицца", "food", 0.7, (0.1, 0.1, 0.2, 0.2))
    manager.record(1, food, frame, person)
    assert len(sent) == 1
    with factory() as db:
        assert db.query(ObjectObservation).count() == 2

    # ---- API ----
    with factory() as db:
        rows = objects_api.api_objects_history(db=db)
        assert len(rows) == 2 and rows[0]["camera_name"] == "Касса"
        weapon_row = next(r for r in rows if r["category"] == "weapon")
        assert weapon_row["category_label"] == "оружие"
        assert weapon_row["has_snapshot"] is True
        rows = objects_api.api_objects_history(category="food", db=db)
        assert len(rows) == 1 and rows[0]["label"] == "pizza"

    class FakeState:
        def snapshot(self):
            return [{
                "camera_id": 1,
                "tracks": [],
                "objects": [{"label": "knife", "label_ru": "нож",
                             "category": "weapon", "confidence": 0.8,
                             "bbox": [0.1, 0.1, 0.2, 0.2], "track_ids": [1]}],
            }]

    request = SimpleNamespace(app=SimpleNamespace(
        state=SimpleNamespace(detection_state=FakeState())))
    with factory() as db:
        live = objects_api.api_objects_live(request=request, db=db)
    assert len(live) == 1 and live[0]["camera_name"] == "Касса"
    assert live[0]["category_label"] == "оружие"
    request_none = SimpleNamespace(
        app=SimpleNamespace(state=SimpleNamespace(detection_state=None)))
    with factory() as db:
        assert objects_api.api_objects_live(request=request_none, db=db) == []


def test_appearance_manager_and_api():
    factory = make_db()
    manager = AppearanceLogManager(factory, keep_days=30)
    manager.record(1, 17, 7, None, "синий", "чёрный", "мужчина", 35)
    with factory() as db:
        assert db.query(AppearanceObservation).count() == 1

    with factory() as db:
        rows = appearance_api.api_appearance_history(db=db)
        assert len(rows) == 1
        assert rows[0]["employee_name"] == "Иван Петров"
        assert rows[0]["upper_color"] == "синий"
        rows = appearance_api.api_appearance_history(color="чёрный", db=db)
        assert len(rows) == 1, "поиск и по нижней одежде"
        rows = appearance_api.api_appearance_history(gender="женщина", db=db)
        assert rows == []

    manager.record(1, 18, None, 5, "красный", None, None, None)
    with factory() as db:
        latest = appearance_api.api_appearance_latest(db=db)
        assert len(latest) == 2
        emp = [r for r in latest if r["employee_id"] == 7][0]
        assert emp["lower_color"] == "чёрный"
        glob = [r for r in latest if r["global_id"] == 5][0]
        assert glob["employee_name"] is None and glob["upper_color"] == "красный"

    # очистка старых записей
    with factory() as db:
        db.add(AppearanceObservation(
            camera_id=1, track_id=1,
            created_at=utcnow() - timedelta(days=40)))
        db.commit()
        total = db.query(AppearanceObservation).count()
    manager.cleanup()
    with factory() as db:
        assert db.query(AppearanceObservation).count() == total - 1


# --------------------------------------------------------------------- main

def test_letterbox_pad_regression():
    """Регрессия: отступ letterbox вычитался и из РАЗМЕРОВ боксов — на
    16:9-потоках (pad_y=140 при 1080p) объекты ниже ~420px получали
    отрицательную высоту и молча отбрасывались (чипсы на столе и т.п.)."""
    from app.ai import detector as det_mod
    from types import SimpleNamespace

    # фейковая ONNX-сессия: один якорь — person cx=320, cy=320, w=60, h=100
    # (letterbox 640), score 0.9; кадр 1920x1080 → scale=1/3, pad=(0, 140)
    out = np.zeros((1, 84, 8400), dtype=np.float32)
    out[0, 0, 0], out[0, 1, 0] = 320.0, 320.0
    out[0, 2, 0], out[0, 3, 0] = 60.0, 100.0
    out[0, 4, 0] = 0.9

    class FakeSession:
        def run(self, *_a, **_k):
            return [out]

        def get_inputs(self):
            return [SimpleNamespace(name="images")]

        def get_providers(self):
            return ["CPUExecutionProvider"]

    orig = det_mod.ort.InferenceSession
    det_mod.ort.InferenceSession = lambda *a, **k: FakeSession()
    try:
        det = det_mod.PersonDetector("fake.onnx", ["CPUExecutionProvider"], 0.35)
    finally:
        det_mod.ort.InferenceSession = orig

    frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
    dets = det.detect(frame)
    assert len(dets) == 1, f"детекция потеряна: {dets}"
    x1, y1, x2, y2 = dets[0][0].tolist()
    # ожидание: x1=(320-30)/⅓=870, y1=(320-50-140)/⅓=390, w=180, h=300
    assert abs(x1 - 870) <= 2 and abs(y1 - 390) <= 2, (x1, y1)
    assert abs(x2 - 1050) <= 2 and abs(y2 - 690) <= 2, (x2, y2)


def test_extra_model_specs_and_forced_category():
    import os
    from app.config import Settings

    s = Settings(extra_models=(
        " models/snacks.onnx | chips bag, snack bag | food | 0.30 ;"
        "models/w.onnx|pistol,rifle|weapon;"
        "models/bad.onnx"
    ))
    specs = s.extra_model_specs()
    assert len(specs) == 2, "запись без классов отбрасывается"
    assert specs[0]["path"] == "models/snacks.onnx"
    assert specs[0]["classes"] == ["chips bag", "snack bag"]
    assert specs[0]["category"] == "food" and specs[0]["confidence"] == 0.30
    assert specs[1]["category"] == "weapon"
    assert specs[1]["confidence"] is None, "порог по умолчанию — OBJECT_CONFIDENCE"

    s2 = Settings(extra_models="models/x.onnx|a,b")
    assert s2.extra_model_specs()[0]["category"] == "other"

    # forced_category: имена кастомной оружейной модели не из WEAPON_LABELS
    raws = [scene_raw(0, PERSON_BBOX), scene_raw(3, KNIFE_BBOX, 0.7)]
    objs = build_scene_objects(raws, lambda cid: ["person", "x", "x", "pistol"][cid],
                               W, H, forced_category="weapon")
    assert [o.category for o in objs] == ["weapon"]
    assert objs[0].label == "pistol"


def main() -> int:
    tests = [
        test_categories_and_colors,
        test_build_scene_objects,
        test_letterbox_pad_regression,
        test_extra_model_specs_and_forced_category,
        test_tracker_min_hits_and_items,
        test_tracker_ttl_and_drop_camera,
        test_recognition_service_split,
        test_gender_age_collection,
        test_attribute_extractor,
        test_object_manager_and_api,
        test_appearance_manager_and_api,
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
