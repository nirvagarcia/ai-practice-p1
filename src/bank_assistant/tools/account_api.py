"""Account lookup port and an in-memory mock adapter."""

from pathlib import Path
from typing import Protocol

from pydantic import TypeAdapter

from bank_assistant.domain.errors import NotFoundError
from bank_assistant.domain.schemas import Account

_ACCOUNTS = TypeAdapter(list[Account])


class AccountGateway(Protocol):
    def get_account(self, account_id: str) -> Account: ...

    def get_account_for_owner(self, account_id: str, owner_id: str) -> Account: ...

    def accounts_of(self, owner_id: str) -> tuple[Account, ...]: ...


class InMemoryAccountGateway:
    def __init__(self, accounts: list[Account]) -> None:
        self._accounts = {account.account_id: account for account in accounts}

    @classmethod
    def from_fixture(cls, fixture: Path) -> "InMemoryAccountGateway":
        return cls(_ACCOUNTS.validate_json(fixture.read_bytes()))

    def get_account(self, account_id: str) -> Account:
        try:
            return self._accounts[account_id]
        except KeyError:
            raise NotFoundError("account", account_id) from None

    def get_account_for_owner(self, account_id: str, owner_id: str) -> Account:
        account = self.get_account(account_id)
        if account.owner_id != owner_id:
            raise NotFoundError("account", account_id) from None
        return account

    def accounts_of(self, owner_id: str) -> tuple[Account, ...]:
        return tuple(a for a in self._accounts.values() if a.owner_id == owner_id)
