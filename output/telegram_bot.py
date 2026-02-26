"""
output/telegram_bot.py
Telegram bot integration for ScoreBorga 2.5.
Sends formatted prediction messages to a configured Telegram chat.
"""

import asyncio
import logging
from typing import Optional

from telegram import Bot
from telegram.error import TelegramError

from config.settings import settings

logger = logging.getLogger(__name__)


async def _send_message_async(
    text: str,
    bot_token: str,
    chat_id: str,
    parse_mode: str = "Markdown",
) -> bool:
    """Async helper to send a Telegram message."""
    bot = Bot(token=bot_token)
    try:
        # Telegram messages have a 4096-character limit; split if needed.
        max_length = 4096
        if len(text) <= max_length:
            await bot.send_message(chat_id=chat_id, text=text, parse_mode=parse_mode)
        else:
            # Split on double newlines to keep logical blocks together
            chunks = _split_message(text, max_length)
            for chunk in chunks:
                await bot.send_message(chat_id=chat_id, text=chunk, parse_mode=parse_mode)
        return True
    except TelegramError as exc:
        logger.error("Telegram error while sending message: %s", exc)
        return False
    finally:
        await bot.close()


def _split_message(text: str, max_length: int) -> list:
    """Split a long message into chunks that respect Telegram's length limit."""
    chunks = []
    while len(text) > max_length:
        split_pos = text.rfind("\n\n", 0, max_length)
        if split_pos == -1:
            split_pos = max_length
        chunks.append(text[:split_pos])
        text = text[split_pos:].lstrip()
    if text:
        chunks.append(text)
    return chunks


def send_message(
    text: str,
    bot_token: Optional[str] = None,
    chat_id: Optional[str] = None,
    parse_mode: str = "Markdown",
) -> bool:
    """
    Send a Telegram message synchronously.

    Args:
        text: The message text (Markdown formatted by default).
        bot_token: Telegram bot token (defaults to settings.TELEGRAM_BOT_TOKEN).
        chat_id: Telegram chat ID (defaults to settings.TELEGRAM_CHAT_ID).
        parse_mode: Telegram parse mode ('Markdown' or 'HTML').

    Returns:
        True if sent successfully, False otherwise.
    """
    token = bot_token or settings.TELEGRAM_BOT_TOKEN
    cid = chat_id or settings.TELEGRAM_CHAT_ID

    if not token or not cid:
        logger.error("Telegram bot token or chat ID not configured.")
        return False

    try:
        return asyncio.run(
            _send_message_async(text, token, cid, parse_mode)
        )
    except Exception as exc:
        logger.error("Unexpected error sending Telegram message: %s", exc)
        return False
