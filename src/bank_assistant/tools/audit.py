"""Minimal audit trail. Records synthetic IDs and outcomes only, never secrets or free text."""

from collections.abc import Callable
from datetime import UTC, datetime

from pydantic import BaseModel, ConfigDict


class AuditEvent(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    actor_id: str
    action: str
    resource_id: str
    outcome: str
    occurred_at: datetime


class InMemoryAuditLog:
    def __init__(self, clock: Callable[[], datetime] = lambda: datetime.now(UTC)) -> None:
        self._events: list[AuditEvent] = []
        self._clock = clock

    def record(self, *, actor_id: str, action: str, resource_id: str, outcome: str) -> AuditEvent:
        event = AuditEvent(
            actor_id=actor_id,
            action=action,
            resource_id=resource_id,
            outcome=outcome,
            occurred_at=self._clock(),
        )
        self._events.append(event)
        return event

    @property
    def events(self) -> tuple[AuditEvent, ...]:
        return tuple(self._events)
