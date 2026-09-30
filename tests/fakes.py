"""Offline stand-ins for provider clients."""

from collections.abc import Callable

from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableLambda
from pydantic import BaseModel

from bank_assistant.state.schemas import AssistantState


def user_state(text: str, customer_id: str = "CUST-0001") -> AssistantState:
    return {"messages": [HumanMessage(text)], "customer_id": customer_id}


class FakeStructuredLLM:
    """Duck-types `BaseChatModel.with_structured_output`; responds from the latest human text."""

    def __init__(self, respond: BaseModel | Callable[[str], BaseModel]) -> None:
        self._respond = respond
        self.calls = 0

    def with_structured_output(self, _schema: type) -> RunnableLambda:
        def invoke(messages: list[tuple[str, str]]) -> BaseModel:
            self.calls += 1
            if isinstance(self._respond, BaseModel):
                return self._respond
            return self._respond(messages[-1][1])

        return RunnableLambda(invoke)
