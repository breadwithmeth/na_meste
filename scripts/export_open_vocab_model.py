"""Экспорт open-vocabulary модели (YOLO-World) в ONNX с нужными классами.

Зачем: в COCO (yolov8n.onnx) нет пачки чипсов, снеков и прочего — модель
физически не знает таких классов. YOLO-World — open-vocabulary детектор:
ему можно задать ЛЮБЫЕ текстовые классы, и после экспорта он превращается
в обычный YOLO-совместимый ONNX (выход 4+N, 8400), который подключается
в приложение через EXTRA_MODELS.

Запуск (один раз, нужен интернет; ultralytics ставится отдельно — в venv
приложения намеренно нет torch):

    # 1) разово поставить инструменты экспорта (можно в тот же venv —
    #    после экспорта можно удалить, инференс их не требует):
    venv/Scripts/pip install ultralytics onnx onnxslim

    # 2) экспортировать:
    venv/Scripts/python scripts/export_open_vocab_model.py ^
        --classes "chips bag,snack bag,cookies,candy bar,soda can" ^
        --out models/snacks.onnx --size s

    # 3) включить в .env (порог ниже COCO — open-vocab менее уверенный):
    # EXTRA_MODELS=models/snacks.onnx|chips bag,snack bag,cookies,candy bar,soda can|food|0.30

Порядок классов в --classes = порядок выходных индексов модели.
"""
import argparse
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_CLASSES = "chips bag,snack bag,cookies,candy bar,soda can"
# доступные размеры: s (быстрая, ~25 МБ), m, l (точнее и медленнее)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="YOLO-World → ONNX с произвольными текстовыми классами")
    parser.add_argument(
        "--classes", default=DEFAULT_CLASSES,
        help=f"классы через запятую (по умолчанию: {DEFAULT_CLASSES})")
    parser.add_argument(
        "--out", default="models/snacks.onnx",
        help="куда сохранить ONNX (относительно корня проекта)")
    parser.add_argument(
        "--size", default="s", choices=["s", "m", "l"],
        help="размер yolov8*-worldv2 (s — быстрее, l — точнее)")
    parser.add_argument(
        "--imgsz", type=int, default=640, help="размер входа ONNX")
    args = parser.parse_args()

    try:
        from ultralytics import YOLO
    except ImportError:
        print("ultralytics не установлена. Один раз выполните:")
        print("  venv/Scripts/pip install ultralytics onnx onnxslim")
        print("(инференс приложения её не требует — можно поставить "
              "временно и удалить)")
        return 1

    names = [c.strip() for c in args.classes.split(",") if c.strip()]
    if not names:
        print("Список классов пуст")
        return 1

    model_name = f"yolov8{args.size}-worldv2.pt"
    print(f"Загружаю {model_name} (скачается при первом запуске)…")
    model = YOLO(model_name)
    print(f"Задаю классы: {names}")
    model.set_classes(names)

    out_path = (BASE_DIR / args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Экспортирую ONNX (imgsz={args.imgsz})…")
    exported = model.export(
        format="onnx", imgsz=args.imgsz, simplify=True, opset=12,
    )
    exported_path = Path(exported)
    if exported_path.resolve() != out_path:
        exported_path.replace(out_path)
    print(f"\nГотово: {out_path}")
    print("Подключите в .env:")
    classes_csv = ",".join(names)
    print(f"  EXTRA_MODELS={args.out}|{classes_csv}|food|0.30")
    print("и перезапустите приложение.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
