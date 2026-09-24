# Camera Monitor — MVP

Локальное web-приложение для подключения к IP-видеорегистраторам **Dahua** по RTSP:
просмотр живого видео с камер и управление их конфигурацией.

Это первый этап системы мониторинга сотрудников. Распознавание лиц/сотрудников
и аналитика присутствия — следующие этапы; RTSP-модуль от них не зависит.

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
    ├── Frame Buffer (2 последних кадра, drop old / keep latest)
    │
    └── MJPEG preview → браузер
```

В будущем к Frame Buffer подключится детектор присутствия (Person Detection →
Tracking → Face Recognition → Employee), не меняя RTSP-модуль.

## Стек

- Python 3.11+, FastAPI, SQLAlchemy, SQLite
- FFmpeg / ffprobe (RTSP over TCP, декодирование)
- OpenCV (JPEG-кодирование кадров для MJPEG)
- HTML/CSS/vanilla JS — без frontend-фреймворков

## Установка

### 1. FFmpeg

**Windows**

```bash
winget install --id=Gyan.FFmpeg -e
```

После установки перезапустите терминал (PATH обновляется только в новых
окнах). Альтернатива: скачать сборку с <https://www.gyan.dev/ffmpeg/builds/>
и положить `ffmpeg.exe`/`ffprobe.exe` в папку `bin/` проекта — приложение
найдёт их автоматически. Можно также указать путь явно в `.env`:

```env
FFMPEG_PATH=C:/ffmpeg/bin/ffmpeg.exe
```

**macOS**

```bash
brew install ffmpeg
```

**Ubuntu / Debian**

```bash
sudo apt update && sudo apt install -y ffmpeg
```

Проверка:

```bash
ffmpeg -version
ffprobe -version
```

### 2. Python-зависимости

```bash
python -m venv venv
source venv/bin/activate        # macOS / Linux
pip install -r requirements.txt
```

Windows:

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Конфигурация

Настройки читаются из `.env` (шаблон — `.env.example`):

```env
APP_HOST=0.0.0.0
APP_PORT=8000
DATABASE_URL=sqlite:///./data/app.db
RTSP_CONNECT_TIMEOUT=10
RTSP_RECONNECT_MAX_DELAY=30
DEFAULT_RTSP_PORT=554
```

Пароли камер хранятся в SQLite (`data/app.db`), а не в `.env` — камер может
быть много. Файл `.env` можно не создавать: подойдут значения по умолчанию.

## Запуск

```bash
python main.py
```

Открыть: **<http://localhost:8000>** (интерактивная документация API —
<http://localhost:8000/docs>).

При запуске приложение читает из базы все включённые камеры и автоматически
подключает их.

## Добавление камеры

RTSP URL формируется автоматически из параметров — вводить его вручную не нужно:

```text
NVR IP: 192.168.1.100   RTSP порт: 554   Логин: admin   Пароль: ********
Канал: 4                Поток: main
```

собирается в:

```text
rtsp://admin:password@192.168.1.100:554/cam/realmonitor?channel=4&subtype=0
```

- `channel` — канал камеры на NVR (нумерация с 1);
- `stream_type=main` → `subtype=0` (основной поток), `sub` → `subtype=1`
  (дополнительный, легче для превью).

Кнопка **«Проверить подключение»** реально открывает RTSP-поток и показывает
разрешение, кодек и FPS — либо сообщение об ошибке (без пароля и URL).

## Проверка подключения к Dahua вручную

Если нужно убедиться, что NVR отдаёт RTSP, вне приложения:

```bash
ffprobe -v error -rtsp_transport tcp -select_streams v:0 \
  -show_entries stream=codec_name,width,height,avg_frame_rate \
  -of json "rtsp://admin:ПАРОЛЬ@192.168.1.100:554/cam/realmonitor?channel=1&subtype=0"
```

Успех — JSON с `codec_name`/`width`/`height`/`avg_frame_rate`.
Типичные причины ошибок: неверный пароль, RTSP отключён в настройках NVR
(веб-интерфейс → Настройки → Сеть → RTSP), канал без камеры.

## curl-примеры для API

```bash
# Список камер (+ live-статусы)
curl http://localhost:8000/api/cameras

# Карточка камеры
curl http://localhost:8000/api/cameras/1

# Добавить камеру
curl -X POST http://localhost:8000/api/cameras \
  -H "Content-Type: application/json" \
  -d '{"name":"Касса 1","nvr_host":"192.168.1.100","rtsp_port":554,
       "username":"admin","password":"ПАРОЛЬ","channel":1,
       "stream_type":"main","enabled":true}'

# Изменить камеру (пустой/отсутствующий пароль = не менять)
curl -X PUT http://localhost:8000/api/cameras/1 \
  -H "Content-Type: application/json" \
  -d '{"name":"Касса 1 (новое имя)","enabled":true}'

# Удалить камеру
curl -X DELETE http://localhost:8000/api/cameras/1

# Проверить подключение сохранённой камеры (реально открывает RTSP)
curl -X POST http://localhost:8000/api/cameras/1/test

# Проверить параметры без сохранения (форма добавления)
curl -X POST http://localhost:8000/api/cameras/test \
  -H "Content-Type: application/json" \
  -d '{"nvr_host":"192.168.1.100","rtsp_port":554,"username":"admin",
       "password":"ПАРОЛЬ","channel":1,"stream_type":"main"}'

# Runtime-статус: ONLINE/CONNECTING/..., FPS, разрешение, кодек, реконнекты
curl http://localhost:8000/api/cameras/1/status

# MJPEG-поток (первые ~3 секунды в файл)
curl -m 3 http://localhost:8000/api/cameras/1/stream -o stream.bin
```

MJPEG можно открыть и напрямую в браузере: `http://localhost:8000/api/cameras/1/stream`.

## Статусы камер

| Статус        | Значение                                             |
|---------------|------------------------------------------------------|
| `ONLINE`      | кадры идут                                           |
| `CONNECTING`  | первичное подключение                                |
| `RECONNECTING`| соединение потеряно, переподключение (backoff 1→30 с)|
| `OFFLINE`     | камера выключена или ридер остановлен                |
| `ERROR`       | неисправимая ошибка (например, не найден FFmpeg)     |

При разрыве соединения ридер переподключается автоматически с exponential
backoff (1, 2, 4, 8, 16, 30, 30, ... сек) — без перезапуска приложения.

## Безопасность

- Пароль **никогда** не возвращается API (только `has_password: true`),
  не выводится в UI, не вставляется в HTML/JS и не пишется в логи
  (RTSP URL в логах маскируется: `rtsp://admin:***@...`).
- Пароль передаётся в ffmpeg как аргумент командной строки, поэтому он виден
  в списке процессов локальной машины — это особенность подхода subprocess.
- В MVP нет авторизации: приложение предназначено для запуска на локальной
  машине или в доверенной локальной сети. Не публикуйте порт в интернет.

## Структура проекта

```text
├── main.py                     # входная точка: python main.py
├── requirements.txt
├── .env.example                # шаблон конфигурации
├── data/app.db                 # SQLite (создаётся автоматически)
├── app/
│   ├── config.py               # настройки (.env)
│   ├── api/cameras.py          # REST API + MJPEG-стрим
│   ├── database/
│   │   ├── database.py         # engine/сессии
│   │   └── models.py           # модель Camera
│   ├── rtsp/
│   │   ├── dahua.py            # сборка RTSP URL + маскировка
│   │   ├── reader.py           # RTSPReader + FrameBuffer + реконнект
│   │   └── manager.py          # менеджер ридеров
│   ├── services/camera_service.py  # CRUD + проверка подключения
│   └── templates/              # index.html, camera.html
└── static/                     # css + vanilla js
```

## Дорожная карта (следующие этапы)

Person Detection → Tracking → Face Recognition → сотрудники, зоны,
аналитика присутствия. RTSP-модуль и FrameBuffer уже готовы к этому:
кадры — numpy BGR, буфер отдаёт только актуальные кадры.
