# Мониторинг камер и присутствия сотрудников

Локальное web-приложение:

- **Камеры**: подключение к IP-видеорегистраторам **Dahua** по RTSP, живое превью (MJPEG)
- **AI**: обнаружение людей (YOLO), трекинг (ByteTrack), распознавание лиц
  сотрудников (InsightFace/ArcFace), сессии присутствия, **распознавание
  действий** (сидит / работает / отдыхает / кушает / идёт / лежит / телефон)
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
            ├── Action Recognition (yolov8n-pose: сидит/работает/отдыхает/…)
            │
            └── Presence Sessions → SQLite
                    │
                    ├── Unknown Events (снимок → SQLite → Telegram)
                    ▼
              Web UI + Telegram-группа
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
| YOLOv8n-pose | `models/yolov8n-pose.onnx` (~13 МБ) | поза (17 ключевых точек) для распознавания действий | скачивается автоматически при первом запуске |
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
  и уверенностью, красный — Unknown, жёлтый пунктир — распознаётся) и чип
  действия внутри бокса (сидит / работает / отдыхает / …)
- **Сотрудники**: список, добавление, страница сотрудника с загрузкой фото
  (шаги: поиск лица → эмбеддинг → сохранение; ошибки: нет лица / несколько
  лиц / лицо слишком маленькое; предупреждение о размытости)
- **Присутствие**: кто сейчас на камерах + история сессий
- **Действия**: кто что делает прямо сейчас (сидит / работает / отдыхает /
  кушает / …), сводка длительностей за период и история наблюдений; на
  странице камеры действие показывается чипом внутри бокса человека
- **Посторонние**: галерея зафиксированных неопознанных людей (снимок,
  камера, время) + каждое событие уходит в Telegram
- **Люди**: глобальные личности (межкамерный трекинг) — список с аватарками
  (последний снимок), фильтрами (статус, камера, поиск по G#id или имени
  сотрудника) и счётчиком фиксаций постороннего; карточка с траекторией
  (цепочка камер), таймлайном наблюдений со снимками, слиянием лишних
  global_id; на странице камеры боксы подписаны `T<трек>·G<global_id>`

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
5. события: `person_seen` (появление на камере), `camera_transition`
   (переход) и `person_lost` (identity истекла по gallery_ttl) — в таблице
   `global_events`, история эмбеддингов и снимки — в `global_observations`
   (persistent identity gallery);
6. **дробление устраняется вручную**: т.к. ложное слияние опаснее лишнего
   ID, фрагменты накапливаются — на странице личности лишний G#id
   вливается в нужный (`POST /api/global-persons/{id}/merge/{source}`):
   наблюдения, события и фиксации посторонних переносятся, галерея
   эмбеддингов и живые привязки треков объединяются;
7. ошибки Re-ID/БД изолированы: RTSP → detection → tracking продолжают
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
При включённой 2.5D-модели топология.json не нужна — навигационный граф
в `/spatial-model` заменяет её.

## 2.5D Spatial World Model (метрическая модель помещения)

**Re-ID отвечает на вопрос «похож ли это на того же человека?», 2.5D-модель —
«мог ли он физически оказаться здесь?».** Оба сигнала используются совместно:
модель добавляет к матчингу пространственные оценки и запрещает физически
невозможные переходы (12 м за 2 секунды, другой этаж без лестницы).

Это опциональный слой НАД существующим пайплайном — YOLO, ByteTracker,
Re-ID и topology.json не меняются. Выключен по умолчанию
(`SPATIAL_MODEL_ENABLED=false` — поведение системы идентично прежнему).

```text
RTSP → YOLO → локальный трекер → bbox
    → foot point (низ-центр bbox ≈ ноги на полу)
    → гомография → мировые X/Y (метры)
    → траектория (EMA, скорость, направление)
    → spatial-оценки матчинга → global_id
    → карта 2.5D / таймлайн / события
```

### Что входит

- **Модель помещения** (метры): этажи (z — высота пола), стены/двери/
  лестницы/лифты (отрезки с высотой), зоны (полигоны), планировка-подложка
  с масштабом и привязкой к мировым координатам;
- **Калибровка камер**: ≥4 известных точек пола «клик по кадру → мировые
  координаты» → `cv2.findHomography` → pixel→world; повторная калибровка
  перезаписывает; зона видимости камеры строится автоматически из проекции
  рамки кадра (или задаётся вручную);
- **Навигационный граф**: узлы (коридор/комната/дверь/лестница/лифт/вход/
  выход/запретная зона) и рёбра; межэтажные переходы — только через
  stairs/elevator; Дейкстра даёт расстояние и минимальное время прохода;
- **Скоринг матчинга** (профиль ТЗ при включённой модели):
  `final = W_REID·reid + W_SPATIAL·spatial + W_TEMPORAL·temporal(физическая)
  + W_DIRECTION·direction` — по умолчанию 0.50/0.25/0.15/0.10. Spatial-данных
  нет (камера не откалибрована, первый трек) — прежние 5 слагаемых;
- **Veto**: расстояние/время физически невозможны, другой этаж без
  stairs/elevator — кандидат отбрасывается с причиной;
- **Предсказание**: позиция + скорость + граф → какие камеры «впереди» и
  когда человек попадёт в их зону видимости (ETA);
- **Карта** `/spatial-model`: top-down (стены, двери, зоны, камеры с FOV и
  зонами видимости, люди с траекториями/скоростью/предсказанием) и режим
  «Изометрия» (этажи по высоте, стены параллелограммами) — без внешних
  библиотек, чистый canvas.

### Включение и настройка

1. Откройте `/spatial-model` → «Редактор»: создайте этаж, при желании
   загрузите планировку и задайте масштаб (2 клика + реальное расстояние в
   метрах), нарисуйте стены/двери, разместите камеры (позиция, поворот, FOV);
2. «Калибровка»: выберите камеру → «Снимок» → кликните ≥4 известных точки
   ПОЛА → мировые координаты вручную или кнопкой «карта» → «Калибровать».
   Кнопка «Сетка 1 м» проецирует мировую сетку на кадр — визуальная
   проверка качества калибровки;
3. «Отладка» (Calibration debug): на живом кадре показываются bbox, foot
   point (жёлтый крестик) и мировые координаты каждого человека — те же
   точки одновременно на карте. Это критично для проверки калибровки;
4. `SPATIAL_MODEL_ENABLED=true` в `.env` — модель начинает влиять на
   матчинг (перезапуск приложения).

### Калибровка по ArUco-маркерам (автоматическая)

Без ручных кликов и с субпиксельной точностью:

1. **«Скачать лист маркеров»** на вкладке «Калибровка» — печатный A4-лист
   (`GET /api/spatial/markers/sheet?count=6&marker_cm=10`). Печать строго
   в масштабе 100%; контрольный отрезок 10 см на листе должен совпасть с
   линейкой. Матовая бумага — глянец бликует;
2. Наклейте маркеры малярным скотчем на пол с разбросом по кадру
   (ближняя/дальняя зона, лево/право), замерьте **центр каждого маркера**
   от нуля координат;
3. «Найти маркеры в кадре» — детектор покажет их на снимке (зелёные
   квадраты); заполните таблицу ID → (X, Y);
4. «Калибровать автоматически» — гомография считается по детектированным
   центрам, ошибка репроекции и зона видимости как при ручной калибровке;
5. **Цепочка без рулетки**: координаты маркеров общие для всех камер —
   откалибруйте одну, и для соседних нажмите «Перенести координаты»:
   калиброванная камера сама «обмерит» маркеры своей гомографией
   (`POST /api/spatial/cameras/{id}/markers/measure`).

Словарь `DICT_4X4_50` (крупные биты — устойчивость к наклонным ракурсам
с высоты ~3 м). Маркеры 10 см при съёмке с 3 м; при очень косых ракурсах
крупнее. Пара маркеров можно оставить постоянными вдоль стен — дрейф
калибровки всегда поправится одной кнопкой.

### API

```bash
# полный мир: этажи с объектами, камеры, узлы и рёбра графа
curl -s localhost:8000/api/spatial/world

# люди сейчас в мировых координатах (для карты, поллинг 1 с)
curl -s localhost:8000/api/spatial/live

# калибровка камеры: точки → гомография (перезапись = перекалибровка)
curl -s -X POST localhost:8000/api/spatial/cameras/1/calibration \
  -H "Content-Type: application/json" \
  -d '{"points":[{"pixel":[421,712],"world":[2.0,3.0]}, ...],
       "resolution":[1920,1080]}'

# история/прогноз/отладка матчей
curl -s localhost:8000/api/spatial/trajectories/184
curl -s localhost:8000/api/spatial/prediction/184
curl -s "localhost:8000/api/spatial/debug/matches?limit=20"
```

Полный список: `/docs` (тег `spatial`) — CRUD этажей/объектов/камер/узлов,
планировки, `coverage`, `observations`.

### Данные и производительность

Мир хранится в SQLite (`spatial_floors`, `spatial_features`,
`spatial_camera_setups`, `spatial_calibrations`, `spatial_nodes`,
`spatial_edges`) и загружается в память; после правок через API живой
инстанс перезагружает мир сам. Мировые позиции людей пишутся в
`spatial_observations` с троттлингом `SPATIAL_OBS_INTERVAL=1` с на трек —
никакой тяжёлой обработки на каждый кадр нет (проекция = умножение 3×3,
Дейкстра кешируется до перезагрузки мира).

### Отладка матчинга

`MC_DEBUG=true` — логи `[MATCH]` с spatial-оценками и причинами veto.
Каждое решение (MATCHED / NEW_IDENTITY + причины отказов) хранится в ring
buffer: `GET /api/spatial/debug/matches`, spatial-оценки также попадают в
payload событий `person_seen`/`camera_transition`.

## Распознавание действий (Action Recognition)

Слой над трекингом отвечает на вопрос «**что делает человек**»: стоит, идёт,
сидит, лежит, работает, отдыхает, ест/пьёт, говорит по телефону. Работает
по всем людям в кадре — и сотрудникам, и посторонним.

```text
трек человека → кроп → yolov8n-pose (17 ключевых точек COCO)
    → геометрия скелета → мгновенное действие
    → голосование по окну ~6 с (гистерезис, как MIN_CONFIRMATIONS у лиц)
    → итоговое действие → DetectionState (UI/API) + action_observations (БД)
```

- **Мгновенные действия** — правила по ключевым точкам: лежит (торс ближе
  к горизонтали), телефон (кисть у уха), ест/пьёт (кисть у рта), сидит
  (бёдра низко в боксе, колени у бёдер, бокс «квадратный»), идёт (заметное
  перемещение, торс-длин/сек), стоит (всё остальное);
- **Производные действия** — по времени: сидит с устойчиво занятыми руками
  в «рабочей зоне» перед корпусом → **работает**; сидит без активности рук
  дольше `ACTION_REST_AFTER` → **отдыхает**; рука к лицу в ≥30% кадров
  (между укусами) → **ест/пьёт**;
- **Устойчивость**: раз в `ACTION_INTERVAL` секунд на трек (не каждый кадр),
  смена действия — только когда новое действие строго перевесило старое
  в окне голосования;
- **Наблюдения пишутся в БД** при смене действия и далее с троттлингом
  `ACTION_LOG_INTERVAL` (грубая длительность = число наблюдений × интервал),
  хранятся `ACTION_KEEP_DAYS`; привязываются к сотруднику и global_id,
  если известны;
- **Отказоустойчивость**: модель скачивается автоматически при первом
  запуске; при любой проблеме слой отключается с предупреждением в логе —
  детекция/трекинг/лица работают как раньше.

Действия видно: чипом внутри бокса на странице камеры, на странице
**«Действия»** (сейчас / сводка за смену / история с фильтром), через API
(`/api/actions/live`, `/api/actions`, `/api/actions/summary`).

Классификация — геометрическая эвристика по позе (не нейросеть действий
по видеоклипам): она дёшева, интерпретируема и настраивается, но «работает»
и «отдыхает» — это интерпретация позы рук и времени, а не семантика сцены.
Для задач типа «сидит за столом» с камерой под потолком точность сидения
выше, чем у низких ракурсов.

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

# Действия (поза → сидит/работает/отдыхает/кушает/…)
ACTION_RECOGNITION_ENABLED=true
ACTION_INTERVAL=1.0           # сек между оценками позы на трек
ACTION_SMOOTH_SECONDS=6.0     # окно голосования действий
ACTION_LOG_INTERVAL=10.0      # сек между записями в БД без смены действия
ACTION_REST_AFTER=15.0        # сек сидения без активности рук → «отдыхает»
ACTION_KEEP_DAYS=30

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

# Действия
curl http://localhost:8000/api/actions/live             # кто что делает сейчас
curl "http://localhost:8000/api/actions?limit=50"       # история наблюдений
curl "http://localhost:8000/api/actions?employee_id=1&action=working"
curl "http://localhost:8000/api/actions/summary?hours=8" # длительности за смену

# Посторонние
curl http://localhost:8000/api/unknown?limit=50          # + global_id каждой фиксации
curl "http://localhost:8000/api/unknown?global_id=184"   # все фиксации конкретного человека
curl http://localhost:8000/api/unknown/1/photo -o person.jpg
curl -X DELETE http://localhost:8000/api/unknown/1

# Межкамерный трекинг
curl http://localhost:8000/api/global-persons
curl "http://localhost:8000/api/global-persons?status=ACTIVE&search=G#184"
curl http://localhost:8000/api/global-persons/1
curl http://localhost:8000/api/global-persons/1/photo          # последний снимок
curl http://localhost:8000/api/global-persons/1/timeline
curl http://localhost:8000/api/global-persons/1/trajectory
curl http://localhost:8000/api/global-persons/1/observations
curl -X POST http://localhost:8000/api/global-persons/1/merge/2  # G#2 вливается в G#1
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
│                               # global_persons, global_observations, global_events,
│                               # action_observations (сидит/работает/отдыхает/…),
│                               # spatial_* (2.5D-модель: этажи, объекты, калибровки,
│                               # граф, мировые наблюдения)
├── models/                     # yolov8n.onnx, yolov8n-pose.onnx, buffalo_l/,
│                               # osnet_x0_25_msmt17.onnx
├── app/
│   ├── config.py               # настройки (.env)
│   ├── api/                    # cameras.py, employees.py, presence.py,
│   │                           # unknown.py, global_persons.py, actions.py,
│   │                           # spatial.py
│   ├── database/               # database.py, models.py
│   ├── rtsp/                   # dahua.py, reader.py, manager.py (без изменений AI)
│   ├── ai/                     # detector.py, tracker.py, face_detector.py,
│   │                           # face_recognition.py, recognition_service.py,
│   │                           # reid.py, global_tracker.py, worker.py,
│   │                           # pose.py, actions.py (распознавание действий),
│   │                           # providers.py
│   ├── spatial/                # 2.5D Spatial World Model: geometry (гомография),
│   │                           # world, trajectory, navigation, matcher, prediction,
│   │                           # service — опциональный слой над трекингом
│   ├── services/               # camera_service, employee_service, presence_service,
│   │                           # unknown_service, action_service, telegram_service
│   └── templates/              # index, camera, employees, employee_detail,
│                               # presence, actions, unknown, global_persons,
│                               # global_person_detail, spatial_model
├── static/                     # css + vanilla js (в т.ч. spatial_*.js для карты)
└── tests/                      # test_global_persons.py, test_spatial.py,
                                  # test_actions.py:
                                  # venv/Scripts/python tests/test_actions.py
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
