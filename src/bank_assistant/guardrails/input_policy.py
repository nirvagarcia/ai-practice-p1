"""Deterministic input size and character policy; not an intent or security classifier."""

from dataclasses import dataclass
from unicodedata import category


@dataclass(frozen=True)
class InputDecision:
    allowed: bool
    text: str
    reason: str | None = None


class InputPolicy:
    def __init__(self, max_chars: int = 4000) -> None:
        if max_chars < 1:
            raise ValueError("max_chars must be positive")
        self._max_chars = max_chars

    def inspect(self, text: str) -> InputDecision:
        normalized = text.strip()
        if not normalized:
            return InputDecision(False, "", "empty")
        if len(text) > self._max_chars:
            return InputDecision(False, "", "too_long")
        if any(category(char) == "Cc" and char not in "\n\r\t" for char in text):
            return InputDecision(False, "", "control_character")
        return InputDecision(True, normalized)
