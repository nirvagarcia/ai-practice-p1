"""Bounded output inspection and redaction for obvious synthetic-secret patterns."""

import re
from dataclasses import dataclass

_API_KEY = re.compile(r"\bsk-[A-Za-z0-9_-]{20,}\b")
_CARD_NUMBER = re.compile(r"\b(?:\d[ -]?){13,19}\b")
_CVV = re.compile(
    r"\b(?:cvv|cvc|security code|código de seguridad)\b\s*(?:is|es|:)?\s*\d{3,4}\b", re.I
)
_BLOCKED = "No puedo entregar esta respuesta porque contiene información no permitida."


@dataclass(frozen=True)
class OutputDecision:
    allowed: bool
    text: str
    redacted: bool = False
    reason: str | None = None


class OutputPolicy:
    def __init__(self, max_chars: int = 4000) -> None:
        if max_chars < 1:
            raise ValueError("max_chars must be positive")
        self._max_chars = max_chars

    def inspect(self, text: str) -> OutputDecision:
        if len(text) > self._max_chars:
            return OutputDecision(False, _BLOCKED, reason="too_long")
        redacted = _API_KEY.sub("[REDACTED]", text)
        redacted = _CVV.sub("[REDACTED]", redacted)
        redacted = _CARD_NUMBER.sub(
            lambda match: "[REDACTED]" if _is_luhn_valid(match.group()) else match.group(), redacted
        )
        return OutputDecision(True, redacted, redacted != text)


def _is_luhn_valid(value: str) -> bool:
    digits = [int(char) for char in value if char.isdigit()]
    if not 13 <= len(digits) <= 19:
        return False
    checksum = sum(
        digit * 2 // 10 + digit * 2 % 10 if (len(digits) - index) % 2 == 0 else digit
        for index, digit in enumerate(digits)
    )
    return checksum % 10 == 0
