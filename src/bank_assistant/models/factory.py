"""Single place where provider clients are constructed."""

from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from openai import OpenAI

from bank_assistant.config.settings import Settings
from bank_assistant.models.config import ModelRole, chat_spec, embedding_spec


def build_chat_model(role: ModelRole, settings: Settings) -> BaseChatModel:
    spec = chat_spec(role, settings)
    return ChatOpenAI(
        model=spec.model_name,
        api_key=settings.require_openai_api_key(),
        temperature=0,
        timeout=settings.llm_timeout_seconds,
        max_tokens=settings.max_tokens_per_request,
    )


def build_embeddings(settings: Settings) -> Embeddings:
    spec = embedding_spec(settings)
    return OpenAIEmbeddings(
        model=spec.model_name,
        api_key=settings.require_openai_api_key(),
        timeout=settings.llm_timeout_seconds,
    )


def build_moderation_client(settings: Settings) -> OpenAI:
    return OpenAI(
        api_key=settings.require_openai_api_key(),
        timeout=settings.llm_timeout_seconds,
    )
