"""Authoritative US market pre-market observation batch pipeline."""

from __future__ import annotations

import argparse
import datetime
import json
from pathlib import Path
import zoneinfo

from reporting.publisher import publish_report_v2
from stock_papi.batch.calendar import TradingCalendarSet
from stock_papi.integrations.market_data.us_calendar import get_us_calendar_documents

NEW_YORK = zoneinfo.ZoneInfo("America/New_York")


def run_us_pre_market(
    root: Path | str,
    target_market_date: datetime.date,
    *,
    now: datetime.datetime | None = None,
) -> Path:
    root = Path(root)
    now = now or datetime.datetime.now(datetime.timezone.utc)

    # 1. Calendar & Date Semantics
    cal_docs = get_us_calendar_documents(target_market_date.year - 1, target_market_date.year + 1)
    calendars = TradingCalendarSet.from_documents(cal_docs)
    if not calendars.is_session(target_market_date):
        raise ValueError(f"Target pre-market date {target_market_date} is not a valid US trading session")

    # 2. Load latest US post-close pointer to bind pre-market base
    post_close_path = root / "publish" / "reports" / "v2" / "latest-US-post_close.json"
    if not post_close_path.exists():
        raise RuntimeError("US pre-market requires published US post-close base")

    post_close_ptr = json.loads(post_close_path.read_text(encoding="utf-8"))
    meta_rel = post_close_ptr["metadata"]
    base_meta = json.loads((root / "publish" / "reports" / "v2" / meta_rel).read_text(encoding="utf-8"))

    doc = {
        "schema_version": 2,
        "kind": "absorb-report",
        "product_mode": "observation",
        "market": "US",
        "report_type": "pre_market",
        "source_market_date": base_meta["source_market_date"],
        "applicable_trading_date": target_market_date.isoformat(),
        "published_at": now.isoformat().replace("+00:00", "Z"),
        "data_as_of": base_meta["data_as_of"],
        "forecast_start_date": target_market_date.isoformat(),
        "forecast_end_date": target_market_date.isoformat(),
        "observation_start_date": base_meta["source_market_date"],
        "observation_end_date": target_market_date.isoformat(),
        "source_manifest": base_meta["source_manifest"],
        "source_manifest_sha256": base_meta["source_manifest_sha256"],
        "model_versions": {},
        "title": f"ABSORB 美股盤前市場觀察報告 ({target_market_date})",
        "summary": [f"美股 {target_market_date} 開盤前市場觀察與前一交易日結構摘要。"],
        "warnings": [],
        "content": base_meta.get("content", {}),
        "content_sha256": base_meta["content_sha256"],
        "prediction_capability": base_meta["prediction_capability"],
    }

    res = publish_report_v2(root, doc)
    print(f"Published US pre-market report: {res}")
    return res


def main() -> None:
    parser = argparse.ArgumentParser(description="Run US pre-market observation batch pipeline")
    parser.add_argument("--root", required=True, help="Data root path")
    parser.add_argument("--target-market-date", required=True, help="Target date YYYY-MM-DD")
    args = parser.parse_args()

    target_date = datetime.date.fromisoformat(args.target_market_date)
    run_us_pre_market(Path(args.root), target_date)


if __name__ == "__main__":
    main()
