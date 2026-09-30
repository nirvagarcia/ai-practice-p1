import pytest
from langchain_openai import ChatOpenAI

from bank_assistant.config.settings import ConfigurationError, Settings
from bank_assistant.models.config import ModelRole, chat_spec
from bank_assistant.models.factory import build_chat_model, build_embeddings


def test_empty_env_values_fall_back_to_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RETRIEVAL_TOP_K", "")
    monkeypatch.setenv("ROUTER_MODEL", "")

    settings = Settings(_env_file=None)

    assert settings.retrieval_top_k == 4
    assert settings.router_model == Settings.model_fields["router_model"].default


def test_env_overrides_role_model(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GRADER_MODEL", "custom-grader")

    assert chat_spec(ModelRole.GRADER, Settings(_env_file=None)).model_name == "custom-grader"


def test_missing_key_fails_only_when_building_a_client(settings: Settings) -> None:
    with pytest.raises(ConfigurationError, match="OPENAI_API_KEY"):
        build_chat_model(ModelRole.ROUTER, settings)
    with pytest.raises(ConfigurationError):
        build_embeddings(settings)


def test_key_is_never_exposed_in_repr(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-secret")

    settings = Settings(_env_file=None)

    assert "sk-test-secret" not in repr(settings)
    assert "sk-test-secret" not in str(settings.model_dump())


def test_factory_builds_configured_chat_model(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("GENERATION_MODEL", "gen-model")

    model = build_chat_model(ModelRole.GENERATION, Settings(_env_file=None))

    assert isinstance(model, ChatOpenAI)
    assert model.model_name == "gen-model"
