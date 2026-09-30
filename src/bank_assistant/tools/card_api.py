"""Card port and an in-memory mock adapter with idempotent blocking.

Idempotency records live in process memory only: they are lost on restart.
"""

from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol

from pydantic import TypeAdapter

from bank_assistant.domain.errors import InvalidStateError, NotFoundError
from bank_assistant.domain.schemas import Card, CardBlockResult, CardStatus

_CARDS = TypeAdapter(list[Card])


def _utc_now() -> datetime:
    return datetime.now(UTC)


class CardGateway(Protocol):
    def get_card(self, card_id: str) -> Card: ...

    def block_card(self, card_id: str, idempotency_key: str) -> CardBlockResult: ...


class InMemoryCardGateway:
    def __init__(self, cards: list[Card], clock: Callable[[], datetime] = _utc_now) -> None:
        self._cards = {card.card_id: card for card in cards}
        self._processed: dict[str, CardBlockResult] = {}
        self._clock = clock

    @classmethod
    def from_fixture(
        cls, fixture: Path, clock: Callable[[], datetime] = _utc_now
    ) -> "InMemoryCardGateway":
        return cls(_CARDS.validate_json(fixture.read_bytes()), clock)

    def get_card(self, card_id: str) -> Card:
        try:
            return self._cards[card_id]
        except KeyError:
            raise NotFoundError("card", card_id) from None

    def block_card(self, card_id: str, idempotency_key: str) -> CardBlockResult:
        if previous := self._processed.get(idempotency_key):
            if previous.card_id != card_id:
                raise InvalidStateError("idempotency key already used for another card")
            return previous

        card = self.get_card(card_id)
        if card.status is CardStatus.BLOCKED:
            raise InvalidStateError(f"card '{card_id}' is already blocked")

        self._cards[card_id] = card.model_copy(update={"status": CardStatus.BLOCKED})
        result = CardBlockResult(
            card_id=card_id,
            status=CardStatus.BLOCKED,
            idempotency_key=idempotency_key,
            processed_at=self._clock(),
        )
        self._processed[idempotency_key] = result
        return result
