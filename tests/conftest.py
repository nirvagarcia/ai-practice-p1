from pathlib import Path

import pytest

from bank_assistant.config.settings import Settings

DATA_DIR = Path(__file__).resolve().parents[1] / "data"


@pytest.fixture
def mocks_dir() -> Path:
    return DATA_DIR / "mocks"


@pytest.fixture
def knowledge_base_dir() -> Path:
    return DATA_DIR / "knowledge_base"


@pytest.fixture
def eval_dir() -> Path:
    return DATA_DIR / "eval"


@pytest.fixture
def settings(monkeypatch: pytest.MonkeyPatch) -> Settings:
    """Settings isolated from the developer's `.env` and shell."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    return Settings(_env_file=None)
