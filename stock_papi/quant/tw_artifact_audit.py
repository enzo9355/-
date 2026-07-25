"""Fail-closed inventory audit for existing TW per-symbol history artifacts."""

from __future__ import annotations

import datetime as _datetime
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Iterable, Mapping

from stock_papi.quant.tw_incremental import (
    IncrementalHistoryError,
    load_incremental_artifact,
)


@dataclass(frozen=True)
class ArtifactDateAudit:
    latest_by_symbol: Mapping[str, _datetime.date]
    unavailable_symbols: tuple[str, ...]
    earliest_latest_date: _datetime.date | None
    latest_date_counts: Mapping[str, int]

    @property
    def available_count(self) -> int:
        return len(self.latest_by_symbol)


def audit_artifact_dates(
    root: Path,
    symbols: Iterable[str],
    *,
    target_date: _datetime.date,
) -> ArtifactDateAudit:
    if (
        not isinstance(target_date, _datetime.date)
        or isinstance(target_date, _datetime.datetime)
    ):
        raise TypeError("target_date must be a date")
    latest_by_symbol: dict[str, _datetime.date] = {}
    unavailable = []
    counts: dict[str, int] = {}
    for raw_symbol in symbols:
        symbol = str(raw_symbol)
        try:
            artifact = load_incremental_artifact(root, symbol)
        except IncrementalHistoryError:
            unavailable.append(symbol)
            continue
        if artifact.latest_date > target_date:
            raise IncrementalHistoryError(
                f"historical artifact is newer than target for TW:{symbol}"
            )
        latest_by_symbol[symbol] = artifact.latest_date
        key = artifact.latest_date.isoformat()
        counts[key] = counts.get(key, 0) + 1
    earliest = min(latest_by_symbol.values()) if latest_by_symbol else None
    return ArtifactDateAudit(
        latest_by_symbol=MappingProxyType(latest_by_symbol),
        unavailable_symbols=tuple(sorted(set(unavailable))),
        earliest_latest_date=earliest,
        latest_date_counts=MappingProxyType(dict(sorted(counts.items()))),
    )
