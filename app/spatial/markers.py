"""ArUco-маркеры для авто-калибровки камер (ТЗ: калибровка без ручных кликов).

Словарь DICT_4X4_50 — минимальный битовый размер (самые крупные биты при
данном физическом размере → устойчивее всего к наклонным ракурсам с высоты
~3 м). 50 уникальных ID достаточно для листа калибровки.

Точка калибровки = ЦЕНТР маркера: маркер лежит на полу, замеряется
положение его центра.

Два формата: сетка маркеров на одном листе A4 (PNG) или по одному
крупному маркеру на страницу (многостраничный PDF, собирается вручную
без внешних зависимостей — FlateDecode/zlib).
"""
import math
import zlib
from dataclasses import dataclass

import cv2
import numpy as np

DICT_NAME = "DICT_4X4_50"


@dataclass(frozen=True)
class DetectedMarker:
    marker_id: int
    pixel: tuple[float, float]        # центр в пикселях кадра
    size_px: float                    # средняя сторона в пикселях


def _dictionary() -> object:
    return cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)


def detect_markers(frame: np.ndarray) -> list[DetectedMarker]:
    """Найти все ArUco-маркеры в кадре (новый API 4.7+/5.x, старый — фолбэк)."""
    dictionary = _dictionary()
    if hasattr(cv2.aruco, "ArucoDetector"):
        detector = cv2.aruco.ArucoDetector(dictionary)
        corners, ids, _rejected = detector.detectMarkers(frame)
    else:                                   # opencv < 4.7 (contrib)
        corners, ids, _rejected, _ = cv2.aruco.detectMarkers(frame, dictionary)
    out: list[DetectedMarker] = []
    if ids is None:
        return out
    for marker_id, quad in zip(ids.flatten(), corners):
        pts = quad[0]
        cx = float(np.mean(pts[:, 0]))
        cy = float(np.mean(pts[:, 1]))
        size = float(np.mean([
            np.linalg.norm(pts[j] - pts[(j + 1) % 4]) for j in range(4)
        ]))
        out.append(DetectedMarker(int(marker_id), (cx, cy), size))
    return out


# -------------------------------------------------------- печатный лист

# A4 портрет, см; поля и шапка
PAGE_W_CM, PAGE_H_CM = 21.0, 29.7
MARGIN_CM = 0.8
HEADER_CM = 2.6
LABEL_CM = 1.4          # подпись ID под маркером
GAP_CM = 0.4            # зазор между маркерами
MIN_MARKER_CM = 5.0     # мельче — детект с 3 м ненадёжен


def _fit_layout(count: int, marker_cm: float):
    """Подобрать сетку (cols, rows) под количество маркеров заданного размера.
    Возвращает (cols, rows, итоговый_размер_маркера) — размер может быть
    уменьшен, чтобы лист поместился на одну A4."""
    avail_w = PAGE_W_CM - 2 * MARGIN_CM
    avail_h = PAGE_H_CM - 2 * MARGIN_CM - HEADER_CM
    best = None
    for cols in (1, 2, 3, 4):
        rows = math.ceil(count / cols)
        cell_w = avail_w / cols
        cell_h = avail_h / rows
        fits = min(cell_w - GAP_CM, cell_h - LABEL_CM - GAP_CM)
        size = min(marker_cm, fits)
        if size < MIN_MARKER_CM:
            continue
        if best is None or size > best[2]:
            best = (cols, rows, size)
    if best is None:
        raise ValueError(
            f"{count} маркеров по {marker_cm} см не помещаются на A4 — "
            f"уменьшите размер или количество")
    return best


def generate_sheet(count: int = 6, marker_cm: float = 10.0,
                   dpi: int = 150) -> bytes:
    """Печатный лист A4 (PNG): сетка ArUco-маркеров с подписями ID,
    инструкция и контрольный отрезок 10 см для проверки масштаба печати.

    Печать строго в масштабе 100% («реальный размер»), без полей.
    """
    if not 1 <= count <= 50:
        raise ValueError("Количество маркеров: 1..50 (словарь DICT_4X4_50)")
    cols, rows, marker_cm = _fit_layout(count, marker_cm)
    px_per_cm = dpi / 2.54
    page_w = int(PAGE_W_CM * px_per_cm)
    page_h = int(PAGE_H_CM * px_per_cm)
    page = np.full((page_h, page_w, 3), 255, dtype=np.uint8)

    font = cv2.FONT_HERSHEY_SIMPLEX
    ink = (20, 20, 20)

    # шапка-инструкция (латиницей — putText не умеет кириллицу)
    y = int(0.5 * px_per_cm)
    for line in (
        "PALEVO - ArUco markers for camera calibration",
        "PRINT AT 100% SCALE (ACTUAL SIZE), then check the 10 cm bar below",
        "Glue markers to the floor; measure each marker CENTER from (0,0)",
    ):
        cv2.putText(page, line, (int(MARGIN_CM * px_per_cm), y), font, 0.52,
                    ink, 1, cv2.LINE_AA)
        y += int(0.62 * px_per_cm)

    marker_px = int(round(marker_cm * px_per_cm))
    avail_w = PAGE_W_CM - 2 * MARGIN_CM
    avail_h = PAGE_H_CM - 2 * MARGIN_CM - HEADER_CM
    cell_w = avail_w / cols
    cell_h = avail_h / rows

    dictionary = _dictionary()
    for i in range(count):
        col, row = i % cols, i // cols
        # центр ячейки
        cx_cm = MARGIN_CM + cell_w * (col + 0.5)
        cy_cm = MARGIN_CM + HEADER_CM + cell_h * (row + 0.55)
        x0 = int(round((cx_cm - marker_cm / 2) * px_per_cm))
        y0 = int(round((cy_cm - marker_cm / 2) * px_per_cm))
        img = cv2.aruco.generateImageMarker(dictionary, i, marker_px)
        page[y0:y0 + marker_px, x0:x0 + marker_px] = \
            cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        # подпись под маркером
        label = f"ID {i}"
        (tw, th), _ = cv2.getTextSize(label, font, 0.7, 2)
        lx = int(cx_cm * px_per_cm) - tw // 2
        ly = y0 + marker_px + int(0.75 * px_per_cm)
        cv2.putText(page, label, (lx, ly), font, 0.7, ink, 2, cv2.LINE_AA)

    # контрольный отрезок 10 см — проверка, что печать не масштабировалась
    bar_y = page_h - int(0.9 * px_per_cm)
    bar_x0 = int(MARGIN_CM * px_per_cm)
    bar_x1 = int((MARGIN_CM + 10.0) * px_per_cm)
    cv2.line(page, (bar_x0, bar_y), (bar_x1, bar_y), ink, 3)
    cv2.line(page, (bar_x0, bar_y - 8), (bar_x0, bar_y + 8), ink, 2)
    cv2.line(page, (bar_x1, bar_y - 8), (bar_x1, bar_y + 8), ink, 2)
    cv2.putText(page, "10 cm - must match a ruler exactly", (bar_x1 + 10, bar_y + 6),
                font, 0.5, ink, 1, cv2.LINE_AA)

    ok, buf = cv2.imencode(".png", page, [cv2.IMWRITE_PNG_COMPRESSION, 6])
    if not ok:
        raise RuntimeError("Не удалось закодировать лист маркеров в PNG")
    return buf.tobytes()


# ------------------------------------------- один маркер на страницу (PDF)

def render_marker_pages(count: int = 20, marker_cm: float = 18.0,
                        dpi: int = 150) -> list[np.ndarray]:
    """Страницы A4 (RGB): по одному крупному маркеру с ID и контрольным
    отрезком 10 см на каждой."""
    if not 1 <= count <= 50:
        raise ValueError("Количество маркеров: 1..50 (словарь DICT_4X4_50)")
    avail_w = PAGE_W_CM - 2 * MARGIN_CM
    avail_h = PAGE_H_CM - 2 * MARGIN_CM - HEADER_CM - LABEL_CM
    if marker_cm + 0.6 > avail_w or marker_cm + 0.6 > avail_h:
        raise ValueError(
            f"Маркер {marker_cm} см не помещается на A4 — максимум "
            f"{min(avail_w, avail_h) - 0.6:.1f} см")
    px_per_cm = dpi / 2.54
    page_w = int(PAGE_W_CM * px_per_cm)
    page_h = int(PAGE_H_CM * px_per_cm)
    marker_px = int(round(marker_cm * px_per_cm))
    font = cv2.FONT_HERSHEY_SIMPLEX
    ink = (20, 20, 20)
    dictionary = _dictionary()
    pages = []
    for marker_id in range(count):
        page = np.full((page_h, page_w, 3), 255, dtype=np.uint8)
        # шапка (латиницей — putText не умеет кириллицу)
        y = int(0.5 * px_per_cm)
        for line in (
            f"PALEVO - ArUco marker ID {marker_id} (DICT_4X4_50)",
            "PRINT AT 100% SCALE (ACTUAL SIZE), check the 10 cm bar below",
            "Glue to the floor; measure the marker CENTER from (0,0)",
        ):
            cv2.putText(page, line, (int(MARGIN_CM * px_per_cm), y), font,
                        0.52, ink, 1, cv2.LINE_AA)
            y += int(0.62 * px_per_cm)
        # маркер по центру (чуть выше середины, под ним подпись)
        x0 = (page_w - marker_px) // 2
        y0 = int((HEADER_CM + 0.8) * px_per_cm)
        img = cv2.aruco.generateImageMarker(dictionary, marker_id, marker_px)
        page[y0:y0 + marker_px, x0:x0 + marker_px] = \
            cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        label = f"ID {marker_id}"
        (tw, th), _ = cv2.getTextSize(label, font, 1.2, 3)
        cv2.putText(page, label,
                    (page_w // 2 - tw // 2,
                     y0 + marker_px + int(1.0 * px_per_cm)),
                    font, 1.2, ink, 3, cv2.LINE_AA)
        # контрольный отрезок 10 см на каждой странице
        bar_y = page_h - int(0.9 * px_per_cm)
        bar_x0 = int(MARGIN_CM * px_per_cm)
        bar_x1 = int((MARGIN_CM + 10.0) * px_per_cm)
        cv2.line(page, (bar_x0, bar_y), (bar_x1, bar_y), ink, 3)
        cv2.line(page, (bar_x0, bar_y - 8), (bar_x0, bar_y + 8), ink, 2)
        cv2.line(page, (bar_x1, bar_y - 8), (bar_x1, bar_y + 8), ink, 2)
        cv2.putText(page, "10 cm", (bar_x1 + 10, bar_y + 6), font, 0.5,
                    ink, 1, cv2.LINE_AA)
        pages.append(page)
    return pages


def generate_pages_pdf(count: int = 20, marker_cm: float = 18.0,
                       dpi: int = 150) -> bytes:
    """Многостраничный PDF: один маркер на страницу (A4, печать 100%)."""
    pages = render_marker_pages(count=count, marker_cm=marker_cm, dpi=dpi)
    return _images_to_pdf(pages, dpi)


def _images_to_pdf(pages: list[np.ndarray], dpi: int) -> bytes:
    """Собрать RGB-страницы в минимальный PDF (FlateDecode, без библиотек).

    Точки PDF = 72/дюйм; страницы A4 при печати сохраняют физический
    размер, заданный DPI рендера.
    """
    pt = 72.0 / dpi
    n = len(pages)
    out = bytearray(b"%PDF-1.4\n")
    offsets: dict[int, int] = {}

    def add(num: int, body: bytes) -> None:
        offsets[num] = len(out)
        out.extend(f"{num} 0 obj\n".encode())
        out.extend(body)
        out.extend(b"\nendobj\n")

    kids = " ".join(f"{3 + i * 3} 0 R" for i in range(n))
    add(1, b"<< /Type /Catalog /Pages 2 0 R >>")
    add(2, f"<< /Type /Pages /Kids [{kids}] /Count {n} >>".encode())
    for i, img in enumerate(pages):
        h, w = img.shape[:2]
        page_num = 3 + i * 3
        content_num = page_num + 1
        image_num = page_num + 2
        add(page_num, (
            f"<< /Type /Page /Parent 2 0 R "
            f"/MediaBox [0 0 {w * pt:.2f} {h * pt:.2f}] "
            f"/Resources << /XObject << /Im0 {image_num} 0 R >> >> "
            f"/Contents {content_num} 0 R >>").encode())
        content = f"q {w * pt:.2f} 0 0 {h * pt:.2f} 0 0 cm /Im0 Do Q".encode()
        add(content_num,
            f"<< /Length {len(content)} >>\nstream\n".encode()
            + content + b"\nendstream")
        raw = zlib.compress(img.tobytes(), 6)
        add(image_num, (
            f"<< /Type /XObject /Subtype /Image /Width {w} /Height {h} "
            f"/ColorSpace /DeviceRGB /BitsPerComponent 8 "
            f"/Filter /FlateDecode /Length {len(raw)} >>\nstream\n").encode()
            + raw + b"\nendstream")
    xref_pos = len(out)
    total = 2 + n * 3          # catalog + pages + 3 объекта на страницу
    out.extend(f"xref\n0 {total + 1}\n".encode())
    out.extend(b"0000000000 65535 f \n")
    for num in range(1, total + 1):
        out.extend(f"{offsets[num]:010d} 00000 n \n".encode())
    out.extend(f"trailer\n<< /Size {total + 1} /Root 1 0 R >>\n"
               f"startxref\n{xref_pos}\n%%EOF".encode())
    return bytes(out)
