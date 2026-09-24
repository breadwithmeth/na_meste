"""Camera Monitor — MVP мониторинга камер Dahua через RTSP.

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

from app.api.cameras import router as cameras_router
from app.config import BASE_DIR, settings
from app.database.database import SessionLocal, init_db
from app.rtsp.manager import ReaderManager

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
    logger.info(
        "Приложение запущено: http://localhost:%d (API docs: /docs)", settings.app_port
    )
    yield
    # shutdown: корректно останавливаем все ридеры и ffmpeg-процессы
    app.state.manager.stop_all()
    logger.info("Приложение остановлено")


app = FastAPI(title="Camera Monitor", lifespan=lifespan)

app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")
templates = Jinja2Templates(directory=BASE_DIR / "app" / "templates")

app.include_router(cameras_router)


@app.get("/", include_in_schema=False)
async def index(request: Request):
    return templates.TemplateResponse(request, "index.html", {})


@app.get("/cameras/{camera_id}", include_in_schema=False)
async def camera_page(request: Request, camera_id: int):
    return templates.TemplateResponse(
        request, "camera.html", {"camera_id": camera_id}
    )


if __name__ == "__main__":
    uvicorn.run(app, host=settings.app_host, port=settings.app_port)
