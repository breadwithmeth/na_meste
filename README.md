# PALEVO — мониторинг камер и присутствия сотрудников

Локальное web-приложение:

- **Камеры**: подключение к IP-видеорегистраторам **Dahua** по RTSP, живое превью (MJPEG)
- **AI**: обнаружение людей (YOLO), трекинг (ByteTrack), распознавание лиц
  сотрудников (InsightFace/ArcFace), сессии присутствия
- **Посторонние**: фиксация неопознанных людей (снимок + камера + время);
  каждая фиксация связана с глобальной личностью человека (G#id) —
  на странице «Посторонние» видно, сколько раз этого же человека ловили
  и на каких камерах (переход по ссылке на его траекторию)
- **Telegram**: уведомления в группу — посторонний (с фото), сотрудник
  пришёл / ушёл, переход человека между камерами (с фото)
- **Межкамерный трекинг**: один человек = один global_id на всех камерах
  (Person Re-ID OSNet + composite matching + топология камер)

RTSP-модуль не зависит от AI; AI получает кадры из его буфера.

## Архитектура

```text
Dahua NVR
    │  RTSP (over TCP)
    ▼
FFmpeg (subprocess, rawvideo bgr24)
    │
    ▼
RTSP Reader (фоновый поток, авто-реконнект)
    │
    ├── MJPEG preview → браузер (полный FPS)
    │
    └── FrameBuffer (queue=2, drop old / keep latest)
            │
            ▼
       AI Worker (1 поток, AI_FPS на камеру)
            │
            ├── YOLO Person Detection (yolov8n.onnx)
            ├── ByteTrack (треки + grace period)
            ├── Face Detection + Embedding (InsightFace buffalo_l)
            ├── Employee Recognition (косинус, MIN_CONFIRMATIONS)
            │
            └── Presence Sessions → SQLite
                    │
                    ├── Unknown Events (снимок → SQLite → Telegram)
                    ▼
              Web UI (PALEVO) + Telegram-группа
```

## Стек

- Python 3.11+, FastAPI, SQLAlchemy, SQLite (WAL)
- FFmpeg / ffprobe (RTSP over TCP, декодирование)
- OpenCV, numpy
- ONNX Runtime GPU/CPU, InsightFace 2.0 (buffalo_l)
- HTML/CSS/vanilla JS — без frontend-фреймворков

## Установка

### 1. FFmpeg

**Windows**

```bash
winget install --id=Gyan.FFmpeg -e
```

После установки перезапустите терминал. Альтернатива: сборка с
<https://www.gyan.dev/ffmpeg/builds/> в папку `bin/` проекта.

**macOS**: `brew install ffmpeg`  **Ubuntu**: `sudo apt install -y ffmpeg`

### 2. Python-зависимости

```bash
python -m venv venv
venv\Scripts\activate            # Windows
source venv/bin/activate         # macOS / Linux
pip install -r requirements.txt
```

**Важно (GPU)**: пакет `insightface` тянет CPU-сборку onnxruntime и затирает
GPU-версию. После установки выполните один раз:

```bash
pip uninstall -y onnxruntime
pip install --no-deps --force-reinstall onnxruntime-gpu
```

Для работы только на CPU используйте обычный `onnxruntime` вместо `-gpu`
(приложение автоматически переключится, не падая).

### 3. CUDA (Windows, опционально)

onnxruntime-gpu требует CUDA 12/13 + cuDNN 9 DLL. Приложение ищет их
автоматически в порядке:

1. `CUDA_DLL_PATH` из .env;
2. `bin/cuda/` проекта;
3. pip-колёса `nvidia-*` в venv;
4. torch из соседнего окружения (например, ComfyUI);
5. установленный CUDA Toolkit.

Если DLL не найдены — inference идёт на CPU, приложение работает.
При старте в логе: `AI Provider: CUDAExecutionProvider` или `CPUExecutionProvider`.

### 4. Модели (папка `models/`)

| Модель | Файл | Назначение | Откуда |
|---|---|---|---|
| YOLOv8n | `models/yolov8n.onnx` (~12 МБ) | детекция людей | [ultralytics assets](https://github.com/ultralytics/assets/releases) |
| InsightFace buffalo_l | `models/buffalo_l/*.onnx` (~300 МБ) | детекция лиц (SCRFD det_10g) + эмбеддинги (ArcFace w600k_r50) | скачивается автоматически при первом запуске в `models/buffalo_l/` |

YOLO можно скачать вручную:

```bash
curl -L -o models/yolov8n.onnx \
  https://github.com/ultralytics/assets/releases/download/v8.4.0/yolov8n.onnx
```

Модели скачиваются один раз и хранятся локально. Переключение CPU/CUDA —
через `AI_DEVICE=auto|cpu|cuda` в .env.

## Запуск

```bash
python main.py
```

Открыть: **<http://localhost:8000>** (API docs — /docs). При старте подключаются
все включённые камеры из базы и запускается AI-воркер.

## Что видно в интерфейсе

- **Камеры** (дашборд): карточки со статусом и счётчиками — людей / сотрудников /
  неизвестных
- **Камера**: живое видео + bounding boxes поверх (зелёный — сотрудник с именем
  и уверенностью, красный — Unknown, жёлтый пунктир — распознаётся)
- **Сотрудники**: список, добавление, страница сотрудника с загрузкой фото
  (шаги: поиск лица → эмбеддинг → сохранение; ошибки: нет лица / несколько
  лиц / лицо слишком маленькое; предупреждение о размытости)
- **Присутствие**: кто сейчас на камерах + история сессий
- **Посторонние**: галерея зафиксированных неопознанных людей (снимок,
  камера, время) + каждое событие уходит в Telegram
- **Люди**: глобальные личности (межкамерный трекинг) — список, карточка
  с траекторией (цепочка камер) и таймлайном наблюдений со снимками;
  на странице камеры боксы подписаны `T<трек>·G<global_id>`

## Фиксация посторонних и Telegram

Посторонним считается человек, который после двух попыток распознавания
не совпал ни с одним сотрудником (или трекается >15 c без распознавания —
лицо не видно). Для каждого трека — одна фиксация, не чаще
`UNKNOWN_EVENT_COOLDOWN=60` c на камеру (защита от спама при трек-чёрне).

Что делает система при фиксации:

1. вырезает снимок человека из кадра → `unknown_events` (SQLite);
2. отправляет фото с подписью в Telegram-группу;
3. событие видно на странице «Посторонние» (хранится `UNKNOWN_KEEP_DAYS=30` дней).

Уведомления о сотрудниках: «Присутствие началось» и «Сессия закрыта»
(время начала/конца и длительность). Настраивается `TELEGRAM_NOTIFY_*`.

Уведомления о переходах между камерами (межкамерный трекинг): при смене
камеры глобальной личностью в группу приходит фото с подписью
«🔄 G#184 (Иван Петров): «Касса 1» → «Склад», сходство 0.91»
(`TELEGRAM_NOTIFY_GLOBAL_TRANSITION=true`). Оповещение о каждой новой
глобальной личности выключено по умолчанию (`TELEGRAM_NOTIFY_GLOBAL_NEW=false`)
— оно дублирует фиксацию посторонних.

### Настройка Telegram

1. Создайте бота через [@BotFather](https://t.me/BotFather), получите токен.
2. Добавьте бота в группу.
3. **Отправьте в группе сообщение `/start`** (или `@имя_бота привет`) —
   бот видит команды даже в privacy-режиме и по ним определит chat_id.
4. Заполните `.env`:

```env
TELEGRAM_BOT_TOKEN=1234567890:AA...
TELEGRAM_CHAT_ID=          # можно оставить пустым — определится по /start
TELEGRAM_NOTIFY_UNKNOWN=true
TELEGRAM_NOTIFY_PRESENCE=true
```

Если `TELEGRAM_CHAT_ID` пуст, приложение само опрашивает getUpdates, пока
не появится `/start`, и сохраняет chat_id в `data/telegram_chat_id.txt`.
Отправка идёт через очередь в фоновом потоке — проблемы с сетью не влияют
на AI и превью. Токен никогда не логируется.

## Межкамерный трекинг (Multi-Camera Person Tracking)

Один человек, перемещающийся между камерами, получает единый `global_id`
на уровне всей системы. Локальный `track_id` остаётся per-camera.

```text
CAM-01 → track_id=17 ─┐
                     ├→ global_id=184      (Person Re-ID OSNet + composite score)
CAM-02 → track_id=42 ─┘
```

Слой над существующим пайплайном (локальный трекинг не меняется):

1. для устойчивого трека периодически (`REID_EMBEDDING_INTERVAL`, по умолчанию
   2.5 с) считается appearance-эмбеддинг (OSNet x0_25, 512-d, GPU);
2. новый локальный трек матчится с активными глобальными личностями по
   **composite score**:

   `final = W_REID·reid + W_TEMPORAL·время + W_TOPOLOGY·топология
   + W_ASPECT·пропорции + W_FACE·лицо`

   Лицо — дополнительный сигнал: два трека одного сотрудника (распознаны
   лицом) — сильный бонус, разных сотрудников — veto;
3. топология (`topology.json`, см. `topology.json.example`): переход между
   камерами допустим только за `min_seconds..max_seconds`, иначе матч
   отклоняется — «человек физически не мог туда попасть»;
4. **ложное слияние опаснее лишнего ID**: при score ниже порога или
   неоднозначности (два кандидата ближе `GLOBAL_MATCH_MARGIN`) создаётся
   новая личность, сомнительный матч пишется как `ambiguous_match`;
5. события: `person_seen` (появление на камере) и `camera_transition`
   (переход) — в таблице `global_events`, история эмбеддингов и снимки —
   в `global_observations` (persistent identity gallery);
6. ошибки Re-ID/БД изолированы: RTSP → detection → tracking продолжают
   работать (проверено тестами).

Включение/выключение: `MULTI_CAMERA_TRACKING_ENABLED=false` — система
работает ровно как раньше. Отладка: `MC_DEBUG=true` выводит
`[REID]/[MATCH]/[IDENTITY]`-логи.

### Модель Re-ID

`models/osnet_x0_25_msmt17.onnx` (~1 МБ) — скачивается один раз:

```bash
curl -L -o models/osnet_x0_25_msmt17.onnx \
  "https://huggingface.co/anriha/osnet_x0_25_msmt17/resolve/main/osnet_x0_25_msmt17.onnx"
```

### Топология камер

Скопируйте `topology.json.example` → `topology.json`, укажите id камер и
допустимые времена переходов, затем `TOPOLOGY_CONFIG=topology.json` в .env.
Без топологии матчи идут только по Re-ID + времени (нейтральная оценка).

## Как работает распознавание

1. YOLO находит людей; ByteTrack назначает `track_id` (grace period
   `TRACK_LOST_TIMEOUT=3` c — пропавшего ненадолго человека трек не теряет);
2. Для неопознанного трека раз в `FACE_RECOGNITION_RETRY_INTERVAL` c
   вырезается область головы, детектируется лицо, считается эмбеддинг;
3. Косинусное сходство со всеми эмбеддингами сотрудников; при
   `MIN_CONFIRMATIONS=2` подряд совпадений ≥ `MIN_RECOGNITION_CONFIDENCE=0.45`
   трек привязывается к сотруднику (защита от false positives);
4. Пока трек жив — повторное распознавание НЕ выполняется;
5. Подтверждённый сотрудник → presence-сессия (одна на сотрудника на камеру):
   `started_at` при первом подтверждении, `last_seen_at` продлевается,
   `ended_at = last_seen_at` после `PRESENCE_END_TIMEOUT=30` c без наблюдений.

Порог 0.45 подобран под buffalo_l; между разными фото одного человека сходство
обычно 0.35–0.55 — поэтому сотруднику стоит загружать 3–5 фото с разными
углами/освещением. Для более строгого/мягкого распознавания меняйте
`MIN_RECOGNITION_CONFIDENCE`.

## Конфигурация (.env)

Основное (шаблон — `.env.example`):

```env
APP_HOST=0.0.0.0
APP_PORT=8000
DATABASE_URL=sqlite:///./data/app.db
RTSP_CONNECT_TIMEOUT=10
RTSP_RECONNECT_MAX_DELAY=30
DEFAULT_RTSP_PORT=554
PREVIEW_FPS=10

# AI
AI_ENABLED=true
AI_FPS=5                      # кадров/сек на камеру для AI
AI_CAMERAS=                   # "1,3,5" — пусто = все камеры
AI_DEVICE=auto                # auto | cpu | cuda
PERSON_CONFIDENCE=0.35
TRACK_LOST_TIMEOUT=3
MIN_RECOGNITION_CONFIDENCE=0.45
MIN_CONFIRMATIONS=2
FACE_RECOGNITION_RETRY_INTERVAL=2
PRESENCE_END_TIMEOUT=30
FACE_MIN_SIZE=80
CUDA_DLL_PATH=                # путь к CUDA/cuDNN DLL (пусто = авто-поиск)

# Telegram
TELEGRAM_BOT_TOKEN=            # токен от @BotFather
TELEGRAM_CHAT_ID=              # пусто = авто-определение по /start в группе
TELEGRAM_NOTIFY_UNKNOWN=true   # посторонний (с фото)
TELEGRAM_NOTIFY_PRESENCE=true  # сотрудник пришёл / ушёл

# Посторонние
UNKNOWN_EVENT_COOLDOWN=60      # сек между фиксациями на одной камере
UNKNOWN_KEEP_DAYS=30           # сколько дней хранить события
```

Пароли камер хранятся в SQLite, не в .env; через API не отдаются.

## Проверка подключения к Dahua вручную

```bash
ffprobe -v error -rtsp_transport tcp -select_streams v:0 \
  -show_entries stream=codec_name,width,height,avg_frame_rate \
  -of json "rtsp://admin:ПАРОЛЬ@192.168.1.100:554/cam/realmonitor?channel=1&subtype=0"
```

## curl-примеры

```bash
# Камеры
curl http://localhost:8000/api/cameras
curl -X POST http://localhost:8000/api/cameras/1/test
curl http://localhost:8000/api/cameras/1/status
curl http://localhost:8000/api/cameras/1/detections     # текущие люди (треки)
curl -m 3 http://localhost:8000/api/cameras/1/stream -o stream.bin

# Сотрудники
curl http://localhost:8000/api/employees
curl -X POST http://localhost:8000/api/employees -H "Content-Type: application/json" \
  -d '{"name":"Иван Петров","external_id":"10023"}'
curl -X POST http://localhost:8000/api/employees/1/faces -F "files=@photo1.jpg" -F "files=@photo2.jpg"
curl http://localhost:8000/api/employees/1?faces=1

# Присутствие
curl http://localhost:8000/api/presence/active
curl "http://localhost:8000/api/presence?limit=100"
curl http://localhost:8000/api/presence/employee/1

# Посторонние
curl http://localhost:8000/api/unknown?limit=50          # + global_id каждой фиксации
curl "http://localhost:8000/api/unknown?global_id=184"   # все фиксации конкретного человека
curl http://localhost:8000/api/unknown/1/photo -o person.jpg
curl -X DELETE http://localhost:8000/api/unknown/1

# Межкамерный трекинг
curl http://localhost:8000/api/global-persons
curl http://localhost:8000/api/global-persons/1
curl http://localhost:8000/api/global-persons/1/timeline
curl http://localhost:8000/api/global-persons/1/trajectory
curl http://localhost:8000/api/global-persons/1/observations
curl http://localhost:8000/api/cameras/1/tracks
```

## Безопасность

- Пароль камеры никогда не возвращается API, не пишется в логи (URL маскируется
  `rtsp://admin:***@...`), не попадает в HTML/JS.
- Хранятся только эмбеддинги лиц и миниатюры 112×112 для управления в UI;
  исходные фотографии сотрудников не сохраняются.
- В логах нет изображений и эмбеддингов.
- Авторизации нет: приложение — для локальной машины или доверенной сети.

## Структура проекта

```text
├── main.py                     # вход: python main.py
├── requirements.txt
├── .env.example
├── topology.json.example       # топология камер для межкамерного трекинга
├── data/app.db                 # SQLite: cameras, employees, employee_faces,
│                               # presence_sessions, unknown_events,
│                               # global_persons, global_observations, global_events
├── models/                     # yolov8n.onnx, buffalo_l/, osnet_x0_25_msmt17.onnx
├── app/
│   ├── config.py               # настройки (.env)
│   ├── api/                    # cameras.py, employees.py, presence.py,
│   │                           # unknown.py, global_persons.py
│   ├── database/               # database.py, models.py
│   ├── rtsp/                   # dahua.py, reader.py, manager.py (без изменений AI)
│   ├── ai/                     # detector.py, tracker.py, face_detector.py,
│   │                           # face_recognition.py, recognition_service.py,
│   │                           # reid.py, global_tracker.py, worker.py, providers.py
│   ├── services/               # camera_service, employee_service, presence_service,
│   │                           # unknown_service, telegram_service
│   └── templates/              # index, camera, employees, employee_detail,
│                               # presence, unknown, global_persons, global_person_detail
└── static/                     # css + vanilla js
```

## Производительность

- AI работает на GPU (CUDA) при наличии; один AI-воркер обслуживает все камеры
  (архитектура допускает добавление воркеров).
- Кадры берутся из буфера ридера (макс. 2, старые отбрасываются) — AI никогда
  не копит очередь и не тормозит превью.
- 12 камер × AI_FPS=5 требует ~60 инференсов/с — GPU справляется; на CPU
  уменьшите AI_FPS или ограничьте AI_CAMERAS.
- Превью и AI независимы: MJPEG отдаёт полный поток, AI берёт свои 5 кадров/с.

## Дорожная карта (не реализовано)

Контроль рабочего времени, опоздания, зарплаты, смены, зоны, автоматический
вход/выход, cross-camera tracking / ReID между камерами, Telegram,
мобильное приложение, облачная синхронизация.
