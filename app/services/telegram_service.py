"""Telegram-уведомления: очередь + фоновая отправка, авто-определение chat_id.

Токен задаётся в .env и никогда не логируется. Если chat_id не указан,
нотификатор опрашивает getUpdates: достаточно отправить боту /start
(или упомянуть его) в группе — chat_id подхватится и сохранится в
data/telegram_chat_id.txt. Отправка сообщений никогда не блокирует
вызывающий поток (AI-воркер, API) и не роняет приложение.
"""
import json
import logging
import queue
import threading
import time
import urllib.request
from pathlib import Path
from typing import Optional

from app.config import BASE_DIR

logger = logging.getLogger("app.services.telegram")

_API = "https://api.telegram.org/bot{token}/{method}"
_STATE_FILE = BASE_DIR / "data" / "telegram_chat_id.txt"


class TelegramNotifier:
    def __init__(self, token: str, chat_id: str = ""):
        self.token = token.strip()
        self.chat_id = chat_id.strip() or self._load_state()
        self._queue: queue.Queue = queue.Queue(maxsize=200)
        self._stop = threading.Event()
        self._sender: Optional[threading.Thread] = None
        self._warned_full = False

    @property
    def configured(self) -> bool:
        return bool(self.token)

    # ----------------------------------------------------------- жизненный цикл

    def start(self) -> None:
        if not self.configured:
            logger.warning("Telegram: TELEGRAM_BOT_TOKEN не задан — уведомления отключены")
            return
        self._sender = threading.Thread(
            target=self._run_sender, name="telegram-sender", daemon=True
        )
        self._sender.start()
        if self.chat_id:
            logger.info("Telegram: уведомления включены (chat_id=%s)", self.chat_id)
        else:
            logger.info(
                "Telegram: chat_id неизвестен — отправьте боту /start в группе, "
                "он подхватится автоматически"
            )
            threading.Thread(
                target=self._discover_chat_id, name="telegram-discovery", daemon=True
            ).start()

    def stop(self) -> None:
        self._stop.set()

    # ------------------------------------------------------------- отправка

    def send_text(self, text: str) -> None:
        self._enqueue({"kind": "text", "text": text})

    def send_photo(self, jpeg: bytes, caption: str = "") -> None:
        self._enqueue({"kind": "photo", "jpeg": jpeg, "caption": caption})

    def _enqueue(self, item: dict) -> None:
        if not self.configured:
            return
        try:
            self._queue.put_nowait(item)
        except queue.Full:
            if not self._warned_full:
                self._warned_full = True
                logger.warning("Telegram: очередь переполнена — часть сообщений отброшена")

    def _run_sender(self) -> None:
        while not self._stop.is_set():
            try:
                item = self._queue.get(timeout=0.5)
            except queue.Empty:
                continue
            for attempt in range(3):
                try:
                    if not self.chat_id:
                        raise RuntimeError("chat_id ещё не определён")
                    if item["kind"] == "text":
                        self._api("sendMessage", {
                            "chat_id": self.chat_id, "text": item["text"],
                        })
                    else:
                        self._api_photo(item["jpeg"], item["caption"])
                    break
                except Exception as exc:
                    if attempt == 2:
                        logger.warning("Telegram: сообщение не отправлено — %s", self._clean(exc))
                    else:
                        time.sleep(2 * (attempt + 1))

    # ------------------------------------------------------------ Telegram API

    def _api(self, method: str, payload: dict) -> dict:
        request = urllib.request.Request(
            _API.format(token=self.token, method=method),
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=20) as resp:
            return json.loads(resp.read())

    def _api_photo(self, jpeg: bytes, caption: str) -> None:
        boundary = "----palevo7351"
        parts = []
        for name, value in (("chat_id", self.chat_id), ("caption", caption)):
            parts.append(
                f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"'
                f'\r\n\r\n{value}\r\n'.encode("utf-8")
            )
        parts.append(
            f'--{boundary}\r\nContent-Disposition: form-data; name="photo"; '
            f'filename="person.jpg"\r\nContent-Type: image/jpeg\r\n\r\n'.encode()
            + jpeg + b"\r\n"
        )
        parts.append(f"--{boundary}--\r\n".encode())
        request = urllib.request.Request(
            _API.format(token=self.token, method="sendPhoto"),
            data=b"".join(parts),
            headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        )
        with urllib.request.urlopen(request, timeout=30) as resp:
            json.loads(resp.read())

    def _clean(self, exc: Exception) -> str:
        """Убирает токен из текста ошибки перед логированием."""
        return str(exc).replace(self.token, "***")

    # --------------------------------------------------- авто-определение chat_id

    def _discover_chat_id(self) -> None:
        """Опрашивает getUpdates, пока не появится чат: /start или упоминание
        бота в группе (или в личке — как запасной вариант)."""
        offset = 0
        while not self._stop.is_set() and not self.chat_id:
            try:
                data = self._api("getUpdates", {"offset": offset})
                candidates = []  # (приоритет_группы, chat_id, название)
                for update in data.get("result", []):
                    offset = max(offset, update["update_id"] + 1)
                    chat = None
                    if "message" in update:
                        chat = update["message"]["chat"]
                    elif "my_chat_member" in update:
                        chat = update["my_chat_member"]["chat"]
                    elif "channel_post" in update:
                        chat = update["channel_post"]["chat"]
                    if not chat:
                        continue
                    chat_type = chat.get("type")
                    if chat_type in ("group", "supergroup"):
                        candidates.append((0, str(chat["id"]), chat.get("title", "группа")))
                    elif chat_type == "private" and "message" in update:
                        candidates.append((1, str(chat["id"]), "личный чат"))
                if candidates:
                    candidates.sort()
                    _prio, chat_id, title = candidates[0]
                    self._set_chat_id(chat_id, title)
                    return
            except Exception as exc:
                logger.debug("Telegram: ошибка авто-определения chat_id — %s", self._clean(exc))
            self._stop.wait(10)

    def _set_chat_id(self, chat_id: str, title: str) -> None:
        self.chat_id = chat_id
        try:
            _STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
            _STATE_FILE.write_text(chat_id, encoding="utf-8")
        except Exception:
            pass
        logger.info("Telegram: chat_id определён (%s) — уведомления включены", title)

    @staticmethod
    def _load_state() -> str:
        try:
            if _STATE_FILE.is_file():
                return _STATE_FILE.read_text(encoding="utf-8").strip()
        except Exception:
            pass
        return ""
