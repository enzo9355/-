"""Checked-in TWSE calendar evidence used by web freshness classification."""

import hashlib
import json

from stock_papi.batch.calendar import TWSE_CALENDAR_URL


_OFFICIAL_CLOSED_DATES = {
    2026: (
        "2026-01-01",
        "2026-02-12", "2026-02-13", "2026-02-16", "2026-02-17",
        "2026-02-18", "2026-02-19", "2026-02-20", "2026-02-27",
        "2026-04-03", "2026-04-06", "2026-05-01", "2026-06-19",
        "2026-09-25", "2026-09-28", "2026-10-09", "2026-10-26",
        "2026-12-25",
    ),
}


def get_tw_calendar_documents(start_year: int, end_year: int) -> list[dict]:
    documents = []
    for year in range(start_year, end_year + 1):
        dates = _OFFICIAL_CLOSED_DATES.get(year)
        if dates is None:
            continue
        evidence = {
            "source_url": TWSE_CALENDAR_URL,
            "year": year,
            "closed_dates": list(dates),
        }
        source_sha256 = hashlib.sha256(
            json.dumps(evidence, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        documents.append({
            "schema_version": 1,
            "market": "TW",
            "year": year,
            "source_url": TWSE_CALENDAR_URL,
            "fetched_at": "2026-02-12T00:00:00+08:00",
            "source_sha256": source_sha256,
            "valid_from": f"{year}-01-01",
            "valid_to": f"{year}-12-31",
            "closed_dates": list(dates),
            "special_open_dates": [],
            "early_closed_dates": [],
        })
    return documents
