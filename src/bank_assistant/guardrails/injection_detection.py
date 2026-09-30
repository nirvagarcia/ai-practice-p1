"""Advisory prompt-injection heuristics. A signal never grants or denies tool permissions."""

import re
from dataclasses import dataclass

_RULES = {
    "override_instructions": re.compile(
        r"\b(?:ignore|disregard|ignora|olvida)\b.{0,80}\b(?:instructions?|instrucciones|rules|reglas)\b",
        re.IGNORECASE,
    ),
    "reveal_prompt": re.compile(
        r"\b(?:reveal|show|print|muestra|revela|imprime)\b.{0,60}"
        r"\b(?:system prompt|prompt del sistema)\b",
        re.IGNORECASE,
    ),
}


@dataclass(frozen=True)
class InjectionSignal:
    detected: bool
    rule_ids: tuple[str, ...]


def detect_injection(text: str) -> InjectionSignal:
    matches = tuple(name for name, pattern in _RULES.items() if pattern.search(text))
    return InjectionSignal(bool(matches), matches)
