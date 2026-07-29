"""Build report metadata from a verified Observation dashboard."""

import datetime
import hashlib
import json

from stock_papi.batch.observation_products import validate_observation_dashboard


def _canonical(document):
    return json.dumps(
        document,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def build_post_close_observation_metadata(
    dashboard,
    calendars,
    *,
    published_at=None,
):
    snapshot = validate_observation_dashboard(dict(dashboard))
    source_date = datetime.date.fromisoformat(snapshot["observation_as_of"])
    applicable = calendars.next_session(source_date)
    timestamp = published_at or datetime.datetime.fromisoformat(
        snapshot["generated_at"].replace("Z", "+00:00")
    )
    if timestamp.tzinfo is None or timestamp.utcoffset() is None:
        raise ValueError("published_at must be timezone-aware")
    capability = dict(snapshot["prediction_capability"])
    content = {
        "dashboard_sha256": hashlib.sha256(_canonical(snapshot)).hexdigest(),
        "market_observation": snapshot["market_observation"],
        "industry_observations": snapshot["industry_observations"],
        "heatmap": snapshot["heatmap"],
        "stock_events": snapshot["stock_events"],
        "trading_status_observations": snapshot.get(
            "trading_status_observations", []
        ),
        "etf_observations": snapshot["etf_observations"],
        "daily_focus": snapshot["daily_focus"],
        "data_quality": snapshot["data_quality"],
    }
    warnings = []
    if snapshot["data_quality"].get("failure_count", 0):
        warnings.append(
            f"{snapshot['data_quality']['failure_count']} 個來源標的未納入本次觀察"
        )
    return {
        "schema_version": 2,
        "product_mode": "observation",
        "report_type": "post_close",
        "market": "TW",
        "source_market_date": source_date.isoformat(),
        "applicable_trading_date": applicable.isoformat(),
        "published_at": timestamp.astimezone(datetime.timezone.utc)
        .isoformat()
        .replace("+00:00", "Z"),
        "forecast_start_date": applicable.isoformat(),
        "forecast_end_date": applicable.isoformat(),
        "observation_start_date": source_date.isoformat(),
        "observation_end_date": applicable.isoformat(),
        "backtest_as_of": None,
        "data_as_of": source_date.isoformat(),
        "source_manifest": snapshot["source_manifest"],
        "source_manifest_sha256": snapshot["source_manifest_sha256"],
        "model_versions": {},
        "prediction_capability": capability,
        "title": f"{source_date.isoformat()} 盤後市場觀察",
        "summary": [str(value)[:500] for value in snapshot["daily_focus"][:20]],
        "warnings": warnings,
        "content": content,
    }
