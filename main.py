"""Мониторинг камер Dahua (RTSP) и присутствия сотрудников (AI).

Запуск:
    python main.py

Открыть:
    http://localhost:8000
"""
import logging
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from app.ai.face_recognition import EmbeddingStore
from app.ai.providers import provider_label
from app.ai.worker import build_ai_worker, build_enroll_engine
from app.api.actions import router as actions_router
from app.api.cameras import router as cameras_router
from app.api.employees import router as employees_router
from app.api.global_persons import router as global_persons_router
from app.api.presence import router as presence_router
from app.api.spatial import router as spatial_router
from app.api.unknown import router as unknown_router
from app.config import BASE_DIR, settings
from app.database.database import SessionLocal, init_db
from app.rtsp.manager import ReaderManager
from app.services.presence_service import PresenceManager
from app.services.telegram_service import TelegramNotifier
from app.services.unknown_service import UnknownEventsManager

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
)
logger = logging.getLogger("app")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # startup: таблицы + автоподключение всех включённых камер из базы
    init_db()
    app.state.manager = ReaderManager()
    with SessionLocal() as session:
        app.state.manager.startup_from_db(session)

    # --- Telegram-уведомления ---
    app.state.telegram = None
    notifier = None
    if settings.telegram_enabled and settings.telegram_bot_token:
        notifier = TelegramNotifier(settings.telegram_bot_token, settings.telegram_chat_id)
        notifier.start()
        app.state.telegram = notifier
    else:
        logger.info("Telegram: уведомления отключены (TELEGRAM_BOT_TOKEN не задан)")

    # --- AI: сотрудники/лица/присутствие/посторонние ---
    app.state.face_engine = None        # FaceEngine для загрузки фото (det 640)
    app.state.embedding_store = None
    app.state.ai_worker = None
    app.state.detection_state = None
    app.state.unknown_manager = UnknownEventsManager(
        SessionLocal, notifier, settings.unknown_keep_days
    )
    app.state.presence = PresenceManager(
        SessionLocal, settings.presence_end_timeout, notifier=notifier
    )

    try:
        store = EmbeddingStore()
        with SessionLocal() as session:
            store.refresh(session)
        app.state.embedding_store = store
        app.state.face_engine = build_enroll_engine()
        if app.state.face_engine is not None:
            logger.info("AI Provider: %s",
                        provider_label(list(app.state.face_engine.app.models.values())[0].session.get_providers()))
    except Exception:
        logger.exception("AI-инициализация для сотрудников не удалась — "
                         "загрузка фото будет недоступна")

    if settings.ai_enabled:
        worker = build_ai_worker(
            app.state.manager, app.state.presence, app.state.embedding_store,
            unknown_manager=app.state.unknown_manager,
            session_factory=SessionLocal,
            notifier=notifier,
        )
        if worker is not None:
            worker.start()
            app.state.ai_worker = worker
            app.state.detection_state = worker.state
    else:
        logger.info("AI отключён (AI_ENABLED=false)")

    logger.info(
        "Приложение запущено: http://localhost:%d (API docs: /docs)", settings.app_port
    )
    yield
    # shutdown: корректно останавливаем AI-воркер, ридеры и ffmpeg-процессы
    if app.state.ai_worker is not None:
        app.state.ai_worker.stop()
    if app.state.telegram is not None:
        app.state.telegram.stop()
    app.state.manager.stop_all()
    logger.info("Приложение остановлено")


app = FastAPI(title="Мониторинг", lifespan=lifespan)

app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")
templates = Jinja2Templates(directory=BASE_DIR / "app" / "templates")

app.include_router(cameras_router)
app.include_router(employees_router)
app.include_router(presence_router)
app.include_router(unknown_router)
app.include_router(global_persons_router)
app.include_router(actions_router)
app.include_router(spatial_router)


@app.get("/", include_in_schema=False)
async def index(request: Request):
    return templates.TemplateResponse(request, "index.html", {})


@app.get("/cameras/{camera_id}", include_in_schema=False)
async def camera_page(request: Request, camera_id: int):
    return templates.TemplateResponse(
        request, "camera.html", {"camera_id": camera_id}
    )


@app.get("/employees", include_in_schema=False)
async def employees_page(request: Request):
    return templates.TemplateResponse(request, "employees.html", {})


@app.get("/employees/{employee_id}", include_in_schema=False)
async def employee_page(request: Request, employee_id: int):
    return templates.TemplateResponse(
        request, "employee_detail.html", {"employee_id": employee_id}
    )


@app.get("/presence", include_in_schema=False)
async def presence_page(request: Request):
    return templates.TemplateResponse(request, "presence.html", {})


@app.get("/actions", include_in_schema=False)
async def actions_page(request: Request):
    return templates.TemplateResponse(request, "actions.html", {})


@app.get("/unknown", include_in_schema=False)
async def unknown_page(request: Request):
    return templates.TemplateResponse(request, "unknown.html", {})


@app.get("/global", include_in_schema=False)
async def global_persons_page(request: Request):
    return templates.TemplateResponse(request, "global_persons.html", {})


@app.get("/global/{global_id}", include_in_schema=False)
async def global_person_page(request: Request, global_id: int):
    return templates.TemplateResponse(
        request, "global_person_detail.html", {"global_id": global_id}
    )


@app.get("/spatial-model", include_in_schema=False)
async def spatial_model_page(request: Request):
    return templates.TemplateResponse(request, "spatial_model.html", {})


if __name__ == "__main__":
    uvicorn.run(app, host=settings.app_host, port=settings.app_port)
