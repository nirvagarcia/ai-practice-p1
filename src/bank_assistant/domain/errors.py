"""Classified business errors. They signal a decision, never a transient failure: do not retry."""


class DomainError(Exception):
    """Base class for business errors."""


class NotFoundError(DomainError):
    def __init__(self, resource: str, resource_id: str) -> None:
        super().__init__(f"{resource} '{resource_id}' not found")
        self.resource = resource
        self.resource_id = resource_id


class InvalidStateError(DomainError):
    """Operation conflicts with the current state of the resource."""
