"""Quota manager for TEJ research acquisition with configurable safety budgets."""

from __future__ import annotations

import datetime
import json
import logging
from pathlib import Path
from typing import Any, Mapping

from .tej import TEJ_PROVIDER, TejClient, TejError, _canonical, _write_immutable


DEFAULT_SAFETY_RATIO = 0.80
DEFAULT_HARD_MAX_REQUESTS = 500
DEFAULT_HARD_MAX_ROWS = 50_000


class TejQuotaManager:
    """Manages and budgets daily TEJ API usage to prevent trial quota exhaustion."""

    def __init__(
        self,
        root: Path | str,
        *,
        safety_ratio: float = DEFAULT_SAFETY_RATIO,
        hard_max_requests: int = DEFAULT_HARD_MAX_REQUESTS,
        hard_max_rows: int = DEFAULT_HARD_MAX_ROWS,
        logger: logging.Logger | None = None,
    ):
        self.root = Path(root).resolve()
        self.safety_ratio = max(0.1, min(float(safety_ratio), 1.0))
        self.hard_max_requests = max(1, int(hard_max_requests))
        self.hard_max_rows = max(1, int(hard_max_rows))
        self.logger = logger or logging.getLogger(__name__)

        self.budget_requests = int(self.hard_max_requests * self.safety_ratio)
        self.budget_rows = int(self.hard_max_rows * self.safety_ratio)

        self.quota_dir = self.root / "research" / "tej" / "v1"
        self.state_file = self.quota_dir / "quota_state.json"
        self.state: dict[str, Any] = {}

    def _today_str(self) -> str:
        return datetime.datetime.now(datetime.timezone.utc).date().isoformat()

    def load_or_sync_state(self, client: TejClient | None = None) -> dict[str, Any]:
        """Load persistent quota state and synchronize with live server info if client is provided."""
        today = self._today_str()
        existing = {}
        if self.state_file.exists():
            try:
                existing = json.loads(self.state_file.read_text(encoding="utf-8"))
            except Exception as exc:
                self.logger.warning("Failed to parse quota state file: %s", exc)

        server_info = None
        if client and client.enabled and client.api_key:
            discovery = client.discover()
            if discovery.get("status") == "authentication_valid":
                server_info = discovery.get("limits", {})
                if "reqDayLimit" in server_info:
                    self.hard_max_requests = int(server_info["reqDayLimit"])
                    self.budget_requests = int(self.hard_max_requests * self.safety_ratio)
                if "rowsDayLimit" in server_info:
                    self.hard_max_rows = int(server_info["rowsDayLimit"])
                    self.budget_rows = int(self.hard_max_rows * self.safety_ratio)

        if existing.get("date") == today:
            state = existing
        else:
            state = {
                "date": today,
                "provider": TEJ_PROVIDER,
                "requests_used": 0,
                "rows_used": 0,
                "tables_fetched": [],
                "history": existing.get("history", [])[-30:],
            }
            if existing.get("date"):
                state["history"].append({
                    "date": existing["date"],
                    "requests_used": existing.get("requests_used", 0),
                    "rows_used": existing.get("rows_used", 0),
                })

        if server_info:
            server_reqs = server_info.get("todayReqCount")
            server_rows = server_info.get("todayRows")
            if isinstance(server_reqs, int):
                state["requests_used"] = max(state.get("requests_used", 0), server_reqs)
            if isinstance(server_rows, int):
                state["rows_used"] = max(state.get("rows_used", 0), server_rows)
            start_d = server_info.get("startDate")
            end_d = server_info.get("endDate")
            state["startDate"] = str(start_d)[:10] if start_d else None
            state["endDate"] = str(end_d)[:10] if end_d else None

        state["hard_max_requests"] = self.hard_max_requests
        state["hard_max_rows"] = self.hard_max_rows
        state["budget_requests"] = self.budget_requests
        state["budget_rows"] = self.budget_rows
        state["requests_remaining_budget"] = max(0, self.budget_requests - state["requests_used"])
        state["rows_remaining_budget"] = max(0, self.budget_rows - state["rows_used"])
        state["is_exhausted"] = (
            state["requests_remaining_budget"] <= 0 or state["rows_remaining_budget"] <= 0
        )

        self.state = state
        self.save_state()
        return self.state

    def can_consume(self, estimated_requests: int = 1, estimated_rows: int = 100) -> bool:
        """Check if planned consumption is within the safety budget."""
        if not self.state or self.state.get("date") != self._today_str():
            self.load_or_sync_state()

        if self.state.get("is_exhausted"):
            return False

        reqs_after = self.state.get("requests_used", 0) + estimated_requests
        rows_after = self.state.get("rows_used", 0) + estimated_rows

        if reqs_after > self.budget_requests:
            self.logger.warning(
                "Request quota budget exceeded: planned=%d, budget=%d",
                reqs_after,
                self.budget_requests,
            )
            return False

        if rows_after > self.budget_rows:
            self.logger.warning(
                "Row quota budget exceeded: planned=%d, budget=%d",
                rows_after,
                self.budget_rows,
            )
            return False

        return True

    def record_consumption(
        self,
        requests_count: int,
        rows_count: int,
        table: str | None = None,
    ) -> dict[str, Any]:
        """Record actual API consumption and update budget state."""
        if not self.state or self.state.get("date") != self._today_str():
            self.load_or_sync_state()

        self.state["requests_used"] = self.state.get("requests_used", 0) + requests_count
        self.state["rows_used"] = self.state.get("rows_used", 0) + rows_count
        if table and table not in self.state.setdefault("tables_fetched", []):
            self.state["tables_fetched"].append(table)

        self.state["requests_remaining_budget"] = max(
            0, self.budget_requests - self.state["requests_used"]
        )
        self.state["rows_remaining_budget"] = max(
            0, self.budget_rows - self.state["rows_used"]
        )
        self.state["is_exhausted"] = (
            self.state["requests_remaining_budget"] <= 0
            or self.state["rows_remaining_budget"] <= 0
        )
        self.state["last_updated"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        self.save_state()
        return self.state

    def save_state(self) -> None:
        """Persist state to disk safely."""
        self.quota_dir.mkdir(parents=True, exist_ok=True)
        content = json.dumps(self.state, ensure_ascii=False, indent=2)
        temp_file = self.state_file.with_suffix(".tmp")
        temp_file.write_text(content, encoding="utf-8")
        temp_file.replace(self.state_file)

    def summary(self) -> dict[str, Any]:
        """Return a sanitized summary of current quota usage and remaining budget."""
        if not self.state:
            self.load_or_sync_state()
        return {
            "date": self.state.get("date"),
            "provider": TEJ_PROVIDER,
            "startDate": self.state.get("startDate"),
            "endDate": self.state.get("endDate"),
            "hard_limits": {
                "requests": self.hard_max_requests,
                "rows": self.hard_max_rows,
            },
            "budget_limits": {
                "safety_ratio": self.safety_ratio,
                "requests": self.budget_requests,
                "rows": self.budget_rows,
            },
            "used_today": {
                "requests": self.state.get("requests_used", 0),
                "rows": self.state.get("rows_used", 0),
            },
            "remaining_budget_today": {
                "requests": self.state.get("requests_remaining_budget", 0),
                "rows": self.state.get("rows_remaining_budget", 0),
            },
            "is_exhausted": self.state.get("is_exhausted", False),
            "tables_fetched": self.state.get("tables_fetched", []),
        }
