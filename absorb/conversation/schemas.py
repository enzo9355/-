from __future__ import annotations

import dataclasses
import datetime as dt
from typing import Any, Literal


AccessLevel = Literal["public", "authenticated", "admin"]


@dataclasses.dataclass(frozen=True)
class ToolCall:
    name: str
    arguments: dict[str, Any]


@dataclasses.dataclass(frozen=True)
class ToolResult:
    name: str
    ok: bool
    data: dict[str, Any] | None = None
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


@dataclasses.dataclass
class PendingConfirmation:
    action_type: str
    parameters: dict[str, Any]
    nonce: str
    expires_at: dt.datetime
    idempotency_key: str
    status: Literal["pending", "confirmed", "cancelled", "expired"] = "pending"
    schema_version: int = 1


@dataclasses.dataclass
class ConversationContext:
    schema_version: int = 1
    last_page_market: str | None = None
    current_market: str | None = None
    current_entity_type: str | None = None
    current_symbol: str | None = None
    current_industry_id: str | None = None
    current_report_type: str | None = None
    comparison_symbols: tuple[str, ...] = ()
    last_question_type: str | None = None
    last_tool_entities: tuple[str, ...] = ()
    pending_confirmation: PendingConfirmation | None = None
    updated_at: dt.datetime | None = None
    expires_at: dt.datetime | None = None


@dataclasses.dataclass(frozen=True)
class ConversationAnswer:
    text: str
    data_as_of: str | None = None
    data_quality: Literal["available", "partial", "stale", "unavailable"] = "unavailable"
    stale: bool = False
    tools_used: tuple[str, ...] = ()
    action_label: str | None = None
    requires_confirmation: bool = False

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)
