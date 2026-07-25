"""Serve existing TW history plus one verified official target-day row."""

from __future__ import annotations

import datetime as _datetime
import gzip
import hashlib
import io
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from stock_papi.integrations.market_data.tw_official_bulk import OfficialDailySnapshot

MAX_COMPRESSED_BYTES = 5 * 1024 * 1024
MAX_UNCOMPRESSED_BYTES = 20 * 1024 * 1024
SOURCE_MODE = "tw_official_bulk_v1"


class IncrementalHistoryError(RuntimeError):
    pass


@dataclass(frozen=True)
class IncrementalArtifact:
    symbol: str
    document: dict[str, Any]
    compressed_sha256: str


class OfficialCompatFetcher:
    """FinMind-compatible callable backed only by local history and official bulk data."""

    SUPPORTED_DATASETS = {
        "TaiwanStockPrice",
        "TaiwanStockInstitutionalInvestorsBuySell",
        "TaiwanStockMarginPurchaseShortSale",
    }

    def __init__(self, root: Path, snapshot: OfficialDailySnapshot, *, pd: Any):
        self.root = Path(root)
        self.snapshot = snapshot
        self.pd = pd
        self._artifacts: dict[str, IncrementalArtifact] = {}

    def _artifact_path(self, symbol: str) -> Path:
        return self.root / "artifacts" / "stocks" / "TW" / f"{symbol}.json.gz"

    def _load_artifact(self, symbol: str) -> IncrementalArtifact:
        if symbol in self._artifacts:
            return self._artifacts[symbol]
        path = self._artifact_path(symbol)
        try:
            compressed_size = path.stat().st_size
            if not 0 < compressed_size <= MAX_COMPRESSED_BYTES:
                raise ValueError("compressed artifact size")
            compressed = path.read_bytes()
            if len(compressed) != compressed_size:
                raise ValueError("compressed artifact changed")
            with gzip.GzipFile(fileobj=io.BytesIO(compressed), mode="rb") as stream:
                decoded = stream.read(MAX_UNCOMPRESSED_BYTES + 1)
            if not decoded or len(decoded) > MAX_UNCOMPRESSED_BYTES:
                raise ValueError("artifact expansion")
            document = json.loads(decoded.decode("utf-8"))
            if (
                not isinstance(document, dict)
                or document.get("schema_version") != 1
                or document.get("market") != "TW"
                or document.get("symbol") != symbol
                or not isinstance(document.get("daily"), list)
            ):
                raise ValueError("artifact schema")
            _datetime.date.fromisoformat(str(document["as_of"]))
        except (KeyError, OSError, TypeError, UnicodeError, ValueError, gzip.BadGzipFile) as exc:
            raise IncrementalHistoryError(f"historical artifact is unavailable for TW:{symbol}") from exc
        artifact = IncrementalArtifact(
            symbol=symbol,
            document=document,
            compressed_sha256=hashlib.sha256(compressed).hexdigest(),
        )
        self._artifacts[symbol] = artifact
        return artifact

    @staticmethod
    def _date(value: Any) -> _datetime.date:
        try:
            return _datetime.datetime.fromisoformat(str(value).replace("Z", "+00:00")).date()
        except ValueError:
            try:
                return _datetime.date.fromisoformat(str(value)[:10])
            except ValueError as exc:
                raise IncrementalHistoryError("historical row date is invalid") from exc

    @staticmethod
    def _number(row: dict[str, Any], name: str, default: float | None = None) -> float:
        value = row.get(name, default)
        if value is None or isinstance(value, bool):
            raise IncrementalHistoryError(f"historical field is invalid: {name}")
        try:
            number = float(value)
        except (TypeError, ValueError) as exc:
            raise IncrementalHistoryError(f"historical field is invalid: {name}") from exc
        if number != number or number in (float("inf"), float("-inf")):
            raise IncrementalHistoryError(f"historical field is invalid: {name}")
        return number

    def _daily_rows(self, symbol: str, start: _datetime.date, end: _datetime.date) -> list[dict[str, Any]]:
        artifact = self._load_artifact(symbol)
        rows = []
        seen = set()
        latest = None
        for item in artifact.document["daily"]:
            if not isinstance(item, dict):
                raise IncrementalHistoryError(f"historical row is invalid for TW:{symbol}")
            row_date = self._date(item.get("Date"))
            if row_date in seen:
                raise IncrementalHistoryError(f"historical dates are duplicated for TW:{symbol}")
            seen.add(row_date)
            latest = row_date if latest is None or row_date > latest else latest
            if start <= row_date <= end:
                rows.append(dict(item, _date=row_date))
        if latest is None:
            raise IncrementalHistoryError(f"historical rows are empty for TW:{symbol}")
        if latest > self.snapshot.target_date:
            raise IncrementalHistoryError(f"historical artifact is newer than target for TW:{symbol}")
        return sorted(rows, key=lambda row: row["_date"])

    def _official_price(self, symbol: str) -> dict[str, Any] | None:
        row = self.snapshot.price_by_symbol.get(symbol)
        return dict(row) if row is not None else None

    def _official_institutional(self, symbol: str) -> list[dict[str, Any]]:
        return [dict(row) for row in self.snapshot.institutional_by_symbol.get(symbol, ())]

    def _official_margin(self, symbol: str) -> dict[str, Any] | None:
        row = self.snapshot.margin_by_symbol.get(symbol)
        return dict(row) if row is not None else None

    @staticmethod
    def _net_rows(date_text: str, symbol: str, total: float, foreign: float) -> list[dict[str, Any]]:
        remainder = total - foreign
        result = []
        for name, net in (("Foreign", foreign), ("InvestmentTrust", remainder), ("Dealer", 0.0)):
            result.append({
                "date": date_text,
                "stock_id": symbol,
                "name": name,
                "buy": max(net, 0.0),
                "sell": max(-net, 0.0),
            })
        return result

    def _verify_existing_target(self, symbol: str, row: dict[str, Any]) -> None:
        official_price = self._official_price(symbol)
        if official_price is None:
            raise IncrementalHistoryError(f"official target price is missing for TW:{symbol}")
        comparisons = {
            "Open": official_price["open"],
            "High": official_price["max"],
            "Low": official_price["min"],
            "Close": official_price["close"],
            "Volume": official_price["Trading_Volume"],
        }
        for historical_name, official_value in comparisons.items():
            if self._number(row, historical_name) != float(official_value):
                raise IncrementalHistoryError(f"existing target row conflicts with official source for TW:{symbol}")
        institutional = self._official_institutional(symbol)
        expected_total = sum(float(item["buy"]) - float(item["sell"]) for item in institutional)
        expected_foreign = sum(float(item["buy"]) - float(item["sell"]) for item in institutional if item["name"] == "Foreign")
        if institutional:
            if self._number(row, "InstitutionalNet", 0.0) != expected_total or self._number(row, "ForeignNet", 0.0) != expected_foreign:
                raise IncrementalHistoryError(f"existing target chip row conflicts with official source for TW:{symbol}")
        margin = self._official_margin(symbol)
        if margin is not None:
            if (
                self._number(row, "MarginBalance", 0.0) != float(margin["MarginPurchaseTodayBalance"])
                or self._number(row, "ShortBalance", 0.0) != float(margin["ShortSaleTodayBalance"])
            ):
                raise IncrementalHistoryError(f"existing target margin row conflicts with official source for TW:{symbol}")

    def __call__(self, dataset: str, code: str, start_date: str, end_date: str):
        if dataset not in self.SUPPORTED_DATASETS:
            raise IncrementalHistoryError(f"unsupported official compatibility dataset: {dataset}")
        symbol = str(code)
        start = _datetime.date.fromisoformat(start_date)
        end = _datetime.date.fromisoformat(end_date)
        target = self.snapshot.target_date
        historical = self._daily_rows(symbol, start, end)
        target_history = next((row for row in historical if row["_date"] == target), None)
        if target_history is not None:
            self._verify_existing_target(symbol, target_history)

        rows: list[dict[str, Any]] = []
        if dataset == "TaiwanStockPrice":
            for row in historical:
                rows.append({
                    "date": row["_date"].isoformat(),
                    "stock_id": symbol,
                    "open": self._number(row, "Open"),
                    "max": self._number(row, "High"),
                    "min": self._number(row, "Low"),
                    "close": self._number(row, "Close"),
                    "Trading_Volume": self._number(row, "Volume", 0.0),
                })
            if start <= target <= end and target_history is None:
                official = self._official_price(symbol)
                if official is None:
                    raise IncrementalHistoryError(f"official target price is missing for TW:{symbol}")
                rows.append(official)
        elif dataset == "TaiwanStockInstitutionalInvestorsBuySell":
            for row in historical:
                rows.extend(self._net_rows(
                    row["_date"].isoformat(), symbol,
                    self._number(row, "InstitutionalNet", 0.0),
                    self._number(row, "ForeignNet", 0.0),
                ))
            if start <= target <= end and target_history is None:
                rows.extend(self._official_institutional(symbol))
        else:
            for row in historical:
                rows.append({
                    "date": row["_date"].isoformat(),
                    "stock_id": symbol,
                    "MarginPurchaseTodayBalance": self._number(row, "MarginBalance", 0.0),
                    "ShortSaleTodayBalance": self._number(row, "ShortBalance", 0.0),
                })
            if start <= target <= end and target_history is None:
                official = self._official_margin(symbol)
                if official is not None:
                    rows.append(official)
        if not rows:
            return self.pd.DataFrame()
        return self.pd.DataFrame(rows).sort_values("date").reset_index(drop=True)

    def lineage_for(self, symbol: str) -> dict[str, Any]:
        artifact = self._load_artifact(symbol)
        return {
            "source_mode": SOURCE_MODE,
            "source_schema_version": self.snapshot.source_schema_version,
            "target_market_date": self.snapshot.target_date.isoformat(),
            "official_manifest_sha256": self.snapshot.manifest_sha256,
            "historical_artifact_sha256": artifact.compressed_sha256,
            "symbol": symbol,
            "official_price_available": symbol in self.snapshot.price_by_symbol,
            "official_institutional_available": symbol in self.snapshot.institutional_by_symbol,
            "official_margin_available": symbol in self.snapshot.margin_by_symbol,
        }
