"""Per-request preflight token/cost reservation and provider usage collection."""

import math
import threading
from typing import Any, override
from uuid import UUID

from langchain_core.callbacks import UsageMetadataCallbackHandler
from langchain_core.messages import BaseMessage
from langchain_core.outputs import LLMResult

from bank_assistant.config.settings import Settings


class BudgetExceededError(RuntimeError):
    """The next model call could exceed the configured request budget."""


class RequestBudgetCallback(UsageMetadataCallbackHandler):
    raise_error = True

    def __init__(self, settings: Settings) -> None:
        super().__init__()
        self._settings = settings
        self._reserved: dict[UUID, tuple[int, int]] = {}
        self._input_tokens = 0
        self._output_tokens = 0
        self._lock = threading.Lock()

    @override
    def on_chat_model_start(
        self,
        serialized: dict[str, Any],
        messages: list[list[BaseMessage]],
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        del serialized, kwargs
        estimated_input = _estimate_prompt(messages)
        reserved_output = self._settings.max_tokens_per_request
        with self._lock:
            total = (
                self._input_tokens
                + self._output_tokens
                + sum(prompt + output for prompt, output in self._reserved.values())
            )
            projected = total + estimated_input + reserved_output
            if projected > self._settings.max_total_tokens_per_request:
                raise BudgetExceededError("Estimated token budget exceeded")
            if self._settings.max_estimated_cost_usd is not None:
                projected_cost = _estimated_cost(
                    self._settings,
                    self._input_tokens + estimated_input,
                    self._output_tokens + reserved_output,
                    self._reserved.values(),
                )
                if projected_cost > self._settings.max_estimated_cost_usd:
                    raise BudgetExceededError("Estimated cost budget exceeded")
            self._reserved[run_id] = estimated_input, reserved_output

    @override
    def on_llm_end(self, response: LLMResult, *, run_id: UUID, **kwargs: Any) -> None:
        super().on_llm_end(response, **kwargs)
        with self._lock:
            self._reserved.pop(run_id, None)
            try:
                usage = response.generations[0][0].message.usage_metadata
            except (AttributeError, IndexError):
                return
            if usage:
                self._input_tokens += usage.get("input_tokens", 0)
                self._output_tokens += usage.get("output_tokens", 0)

    @override
    def on_llm_error(self, error: BaseException, *, run_id: UUID, **kwargs: Any) -> None:
        del error, kwargs
        with self._lock:
            self._reserved.pop(run_id, None)

    @property
    def input_tokens(self) -> int | None:
        return self._input_tokens if self.usage_metadata else None

    @property
    def output_tokens(self) -> int | None:
        return self._output_tokens if self.usage_metadata else None

    @property
    def estimated_cost_usd(self) -> float | None:
        if not self.usage_metadata or self._settings.input_cost_usd_per_million_tokens is None:
            return None
        return _estimated_cost(self._settings, self._input_tokens, self._output_tokens)


def _estimate_prompt(messages: list[list[BaseMessage]]) -> int:
    byte_count = sum(
        len(str(message.content).encode("utf-8")) for prompt in messages for message in prompt
    )
    return math.ceil(byte_count / 2)


def _estimated_cost(
    settings: Settings,
    input_tokens: int,
    output_tokens: int,
    reserved: Any = (),
) -> float:
    reserved_input = sum(prompt for prompt, _ in reserved)
    reserved_output = sum(output for _, output in reserved)
    input_rate = settings.input_cost_usd_per_million_tokens or 0
    output_rate = settings.output_cost_usd_per_million_tokens or 0
    return (
        (input_tokens + reserved_input) * input_rate
        + (output_tokens + reserved_output) * output_rate
    ) / 1_000_000
