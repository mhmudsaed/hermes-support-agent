"""Transports: Telegram adapter (live Bot API + offline fake)."""

from __future__ import annotations

import json
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass, field


@dataclass
class IncomingMessage:
    chat_id: str
    text: str
    sender: str = ""


class Transport:
    def send(self, chat_id: str, text: str) -> None:
        raise NotImplementedError

    def poll(self, timeout_s: int = 30) -> list[IncomingMessage]:
        raise NotImplementedError


class FakeTransport(Transport):
    """In-memory transport for tests and offline demos."""

    def __init__(self, incoming: list[IncomingMessage] | None = None):
        self.incoming: list[IncomingMessage] = list(incoming or [])
        self.sent: list[dict] = []

    def queue(self, chat_id: str, text: str, sender: str = "user") -> None:
        self.incoming.append(IncomingMessage(chat_id=chat_id, text=text,
                                             sender=sender))

    def send(self, chat_id: str, text: str) -> None:
        self.sent.append({"chat_id": chat_id, "text": text,
                          "at": time.time()})

    def poll(self, timeout_s: int = 30) -> list[IncomingMessage]:
        batch, self.incoming = self.incoming, []
        return batch


class TelegramTransport(Transport):
    """Live Telegram Bot API transport (stdlib HTTP, long-polling)."""

    def __init__(self, bot_token: str, default_chat_id: str = "",
                 rate_limit_s: float = 1.0):
        if not bot_token:
            raise ValueError("TelegramTransport needs a bot token")
        self.bot_token = bot_token
        self.default_chat_id = default_chat_id
        self.rate_limit_s = rate_limit_s
        self._offset = 0
        self._last_send = 0.0

    def _api(self, method: str, payload: dict | None = None,
             query: dict | None = None) -> dict:
        url = f"https://api.telegram.org/bot{self.bot_token}/{method}"
        if query:
            url += "?" + urllib.parse.urlencode(query)
        data = json.dumps(payload).encode() if payload is not None else None
        req = urllib.request.Request(
            url, data=data,
            headers={"Content-Type": "application/json"} if data else {})
        with urllib.request.urlopen(req, timeout=65) as resp:
            body = json.loads(resp.read().decode())
        if not body.get("ok"):
            raise RuntimeError(f"telegram API error: {body}")
        return body["result"]

    def send(self, chat_id: str, text: str) -> None:
        wait = self.rate_limit_s - (time.time() - self._last_send)
        if wait > 0:
            time.sleep(wait)
        # Telegram caps messages at 4096 chars; split on newlines.
        chunk, chunks = "", []
        for line in text.splitlines(keepends=True):
            if len(chunk) + len(line) > 4000:
                chunks.append(chunk)
                chunk = ""
            chunk += line
        chunks.append(chunk)
        for part in chunks:
            if part.strip():
                self._api("sendMessage",
                          {"chat_id": chat_id or self.default_chat_id,
                           "text": part})
        self._last_send = time.time()

    def poll(self, timeout_s: int = 30) -> list[IncomingMessage]:
        updates = self._api("getUpdates", query={
            "offset": self._offset, "timeout": min(timeout_s, 50)})
        out: list[IncomingMessage] = []
        for upd in updates:
            self._offset = max(self._offset, upd["update_id"] + 1)
            msg = upd.get("message") or {}
            text = msg.get("text", "")
            chat = msg.get("chat") or {}
            sender = (msg.get("from") or {}).get("username", "")
            if text:
                out.append(IncomingMessage(chat_id=str(chat.get("id", "")),
                                           text=text, sender=sender))
        return out


@dataclass
class TelegramBot:
    """Poll loop wiring a Transport to the SupportAgent Q&A service."""

    agent: object
    transport: Transport

    def handle_text(self, chat_id: str, text: str) -> str:
        cmd = text.strip()
        if cmd.startswith("/start"):
            return ("Send me a question about the docs and I'll answer with "
                    "sources. If I'm unsure, I'll open a support ticket.")
        if cmd.startswith("/tickets"):
            tickets = self.agent.list_tickets("open")
            if not tickets:
                return "No open tickets."
            lines = [f"{t.id}: {t.question[:80]}" for t in tickets[:10]]
            return "Open tickets:\n" + "\n".join(lines)
        result = self.agent.ask(cmd, contact=f"telegram:{chat_id}")
        return result.text

    def serve_once(self) -> int:
        """Poll once and reply to everything; returns messages handled."""
        handled = 0
        for msg in self.transport.poll():
            reply = self.handle_text(msg.chat_id, msg.text)
            self.transport.send(msg.chat_id, reply)
            handled += 1
        return handled

    def serve_forever(self, poll_s: int = 30) -> None:
        while True:
            self.serve_once()


def make_transport(settings, fake: FakeTransport | None = None) -> Transport:
    if fake is not None or not settings.telegram_bot_token:
        return fake or FakeTransport()
    return TelegramTransport(settings.telegram_bot_token,
                             settings.telegram_chat_id)
