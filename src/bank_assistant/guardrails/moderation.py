"""Optional OpenAI moderation adapter with content-free outcomes."""

from typing import Protocol

from openai import OpenAI
from pydantic import BaseModel, ConfigDict


class ModerationResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    flagged: bool
    error_type: str | None = None


class Moderator(Protocol):
    def check(self, text: str) -> ModerationResult: ...


class OpenAIModerator:
    def __init__(self, client: OpenAI, model: str = "omni-moderation-latest") -> None:
        self._client = client
        self._model = model

    def check(self, text: str) -> ModerationResult:
        try:
            response = self._client.moderations.create(model=self._model, input=text)
        except Exception as error:
            return ModerationResult(flagged=False, error_type=type(error).__name__)
        return ModerationResult(flagged=response.results[0].flagged)
