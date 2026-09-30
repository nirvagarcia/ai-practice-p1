"""Typed, versioned evaluation cases loaded from JSONL."""

from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, ValidationError

from bank_assistant.domain.schemas import CustomerId, Intent
from bank_assistant.state.schemas import Outcome

EVAL_DIR = Path("data/eval")

Split = Literal["dev", "holdout"]
ToolName = Literal["calculate_payment", "get_account", "block_card"]


class EvalCase(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    case_id: str
    dataset_version: str
    split: Split
    tags: tuple[str, ...] = ()


class RagCase(EvalCase):
    question: str
    reference: str
    relevant_chunk_ids: tuple[str, ...]

    @property
    def answerable(self) -> bool:
        return bool(self.relevant_chunk_ids)


class RoutingCase(EvalCase):
    message: str
    expected_intent: Intent


class ToolCase(EvalCase):
    message: str
    expected_tool: ToolName | None
    expected_args: dict[str, Any] = {}


class EndToEndCase(EvalCase):
    message: str
    customer_id: CustomerId
    expected_intent: Intent
    expected_outcome: Outcome


def load_cases[T: EvalCase](path: Path, model: type[T], split: Split | None = None) -> list[T]:
    cases: list[T] = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            cases.append(model.model_validate_json(line))
        except ValidationError as error:
            raise ValueError(f"{path}:{number}: {error}") from None
    return [case for case in cases if split is None or case.split == split]
