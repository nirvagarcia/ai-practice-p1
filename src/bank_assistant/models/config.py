"""Logical model roles mapped to provider-level specs."""

from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict

from bank_assistant.config.settings import Settings


class ModelRole(StrEnum):
    ROUTER = "router"
    GENERATION = "generation"
    GRADER = "grader"


class ModelSpec(BaseModel):
    model_config = ConfigDict(frozen=True)

    provider: Literal["openai"] = "openai"
    model_name: str


def chat_spec(role: ModelRole, settings: Settings) -> ModelSpec:
    model_by_role = {
        ModelRole.ROUTER: settings.router_model,
        ModelRole.GENERATION: settings.generation_model,
        ModelRole.GRADER: settings.grader_model,
    }
    return ModelSpec(model_name=model_by_role[role])


def embedding_spec(settings: Settings) -> ModelSpec:
    return ModelSpec(model_name=settings.embedding_model)
