"""
Telegram notification layer for Quantie.

Lifecycle:
  1. Each allowed user sends /start to the bot.
  2. Bot checks username against ALLOWED_USERNAMES whitelist, stores chat_id.
  3. All subsequent push notifications go to every stored chat_id.
  4. Chat IDs persist to data/telegram_chat_ids.json across restarts.

Requires TELEGRAM_BOT_TOKEN in .env.
"""
from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import Callable, Optional

import structlog
from telegram import ReplyKeyboardMarkup, Update
from telegram.ext import Application, CommandHandler, ContextTypes, MessageHandler, filters

log = structlog.get_logger(__name__)

# Comma-separated Telegram usernames allowed to receive notifications (set in .env).
ALLOWED_USERNAMES: frozenset[str] = frozenset(
    u.strip().lstrip("@") for u in os.getenv("TELEGRAM_ALLOWED_USERNAMES", "").split(",") if u.strip()
)
_CHAT_IDS_FILE = Path("data/telegram_chat_ids.json")


class TelegramNotifier:
    """Push-only Telegram notifications. Call start() before engine runs."""

    _STATUS_BUTTON = "📊 Status"
    _KEYBOARD = ReplyKeyboardMarkup([[_STATUS_BUTTON]], resize_keyboard=True, is_persistent=True)

    def __init__(self, token: str) -> None:
        self._app = Application.builder().token(token).build()
        self._app.add_handler(CommandHandler("start", self._on_start))
        self._app.add_handler(CommandHandler("status", self._on_status))
        self._app.add_handler(
            MessageHandler(filters.TEXT & filters.Regex(f"^{self._STATUS_BUTTON}$"), self._on_status)
        )
        self._chat_ids: set[int] = self._load_chat_ids()
        self._status_callback: Optional[Callable[[], dict]] = None

    def set_status_callback(self, cb: Callable[[], dict]) -> None:
        """Register a callable that returns current engine state for /status."""
        self._status_callback = cb

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        await self._app.initialize()
        await self._app.start()
        await self._app.updater.start_polling(drop_pending_updates=True)
        log.info("telegram.notifier.started", registered_users=len(self._chat_ids))

    async def stop(self) -> None:
        await self._app.updater.stop()
        await self._app.stop()
        await self._app.shutdown()

    # ------------------------------------------------------------------
    # Outbound notifications
    # ------------------------------------------------------------------

    async def notify_startup(self, mode: str, bankroll: float, model_dir: str) -> None:
        text = (
            f"\U0001f7e2 *Quantie started*\n"
            f"Mode: `{mode}`\n"
            f"Bankroll: `${bankroll:.2f}`\n"
            f"Model: `{model_dir}`"
        )
        await self._broadcast(text)

    async def notify_shutdown(self, mode: str, realized_pnl: float, reason: str = "") -> None:
        text = (
            f"\U0001f534 *Quantie stopped*\n"
            f"Mode: `{mode}`\n"
            f"Realized PnL: `{realized_pnl:+.2f} USD`"
        )
        if reason:
            text += f"\nReason: `{reason}`"
        await self._broadcast(text)

    async def notify_circuit_breaker(self, kind: str, detail: str) -> None:
        labels = {
            "session_loss": "\U0001f6d1 Session loss limit hit",
            "losing_streak": "\U0001f4c9 Losing streak",
            "latency": "⚠️ Latency halt",
            "drawdown": "⚠️ Drawdown halt",
            "error_rate": "⚠️ Error rate halt",
        }
        title = labels.get(kind, f"⚠️ Circuit breaker: {kind}")
        await self._broadcast(f"{title}\n`{detail}`")

    async def notify_pnl(self, realized_pnl: float, mode: str, bankroll: float) -> None:
        pct = realized_pnl / bankroll * 100 if bankroll else 0.0
        emoji = "\U0001f4c8" if realized_pnl >= 0 else "\U0001f4c9"
        text = (
            f"{emoji} *Hourly PnL update*\n"
            f"Mode: `{mode}`\n"
            f"Realized PnL: `{realized_pnl:+.2f} USD` ({pct:+.1f}%)"
        )
        await self._broadcast(text)

    async def notify_hard_stop(self, realized_pnl: float, session_start: float) -> None:
        loss_pct = abs(realized_pnl) / session_start * 100 if session_start else 0.0
        text = (
            f"\U0001f6d1 *Hard stop — session loss limit reached*\n"
            f"Loss: `{realized_pnl:+.2f} USD` ({loss_pct:.1f}% of session start)\n"
            f"Bot is shutting down."
        )
        await self._broadcast(text)

    # ------------------------------------------------------------------
    # Command handlers
    # ------------------------------------------------------------------

    async def _on_status(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        username = (update.effective_user.username or "").lstrip("@")
        if username not in ALLOWED_USERNAMES:
            await update.message.reply_text("Unauthorized.")
            return
        if self._status_callback is None:
            await update.message.reply_text("Bot not running.")
            return
        try:
            s = self._status_callback()
        except Exception as exc:
            await update.message.reply_text(f"Error fetching status: {exc}")
            return
        pnl = s.get("realized_pnl", 0.0)
        bankroll = s.get("bankroll", 0.0)
        pct = pnl / bankroll * 100 if bankroll else 0.0
        pnl_emoji = "\U0001f4c8" if pnl >= 0 else "\U0001f4c9"
        pending_count = s.get("pending_resolution_count", 0)
        pending_usd = s.get("pending_resolution_usd", 0.0)
        pending_line = (
            f"Pending resolut: `{pending_count} pos  ${pending_usd:.2f}`\n"
            if pending_count > 0 else ""
        )
        text = (
            f"{pnl_emoji} *Quantie Status*\n"
            f"Mode: `{s.get('mode', '?')}`\n"
            f"Balance (total): `${s.get('balance_total', 0.0):.2f}`\n"
            f"Balance (cash):  `${s.get('balance_cash', 0.0):.2f}`\n"
            f"Realized PnL:    `{pnl:+.2f} USD` ({pct:+.1f}%)\n"
            f"Window PnL:      `{s.get('window_pnl', 0.0):+.2f} USD`\n"
            f"{pending_line}"
            f"Status:          `{s.get('status', 'unknown')}`"
        )
        await update.message.reply_text(text, parse_mode="Markdown")

    async def _on_start(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        username = (update.effective_user.username or "").lstrip("@")
        chat_id = update.effective_chat.id
        if username not in ALLOWED_USERNAMES:
            await update.message.reply_text("Unauthorized.")
            log.warning("telegram.unauthorized_start", username=username)
            return
        if chat_id not in self._chat_ids:
            self._chat_ids.add(chat_id)
            self._save_chat_ids()
            log.info("telegram.registered", username=username, chat_id=chat_id)
        await update.message.reply_text(
            f"Registered @{username}. You will receive Quantie notifications here.",
            reply_markup=self._KEYBOARD,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _broadcast(self, text: str) -> None:
        if not self._chat_ids:
            log.debug("telegram.broadcast.no_recipients")
            return
        for chat_id in list(self._chat_ids):
            try:
                await self._app.bot.send_message(
                    chat_id=chat_id,
                    text=text,
                    parse_mode="Markdown",
                )
            except Exception as exc:
                log.warning("telegram.send.failed", chat_id=chat_id, error=str(exc))

    def _load_chat_ids(self) -> set[int]:
        try:
            if _CHAT_IDS_FILE.exists():
                return set(json.loads(_CHAT_IDS_FILE.read_text()))
        except Exception as exc:
            log.warning("telegram.chat_ids.load_failed", error=str(exc))
        return set()

    def _save_chat_ids(self) -> None:
        try:
            _CHAT_IDS_FILE.parent.mkdir(parents=True, exist_ok=True)
            _CHAT_IDS_FILE.write_text(json.dumps(list(self._chat_ids)))
        except Exception as exc:
            log.warning("telegram.chat_ids.save_failed", error=str(exc))


def build_notifier() -> TelegramNotifier | None:
    """Return a TelegramNotifier if TELEGRAM_BOT_TOKEN is set, else None."""
    token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
    if not token:
        log.info("telegram.disabled", reason="TELEGRAM_BOT_TOKEN not set")
        return None
    return TelegramNotifier(token)
