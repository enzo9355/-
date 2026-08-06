"""TPEx lifecycle cache regression tests."""
import datetime
import gzip
import hashlib
import io
import json
import tempfile
import unittest
import uuid
from pathlib import Path
from unittest.mock import patch, Mock

from stock_papi.integrations.market_data.tw_trading_status import (
    HistoricalLifecycleUnavailable,
    _extract_current_mode_effective_date,
    _tpex_mode_events,
    STATUS_PARSER_VERSION,
)
from stock_papi.integrations.market_data.tw_official_cache import (
    OfficialRawCacheEntry,
    load_cached_raw_source,
    store_cached_raw_source,
)

LIFECYCLE_PARSER_VERSION = "tw-lifecycle-parser-v2"


def _roc_date_text(year, month, day):
    return str(year * 10000 + month * 100 + day)


def _tpex_payload(effective_date: datetime.date):
    return json.dumps([
        {
            "Date": _roc_date_text(
                effective_date.year - 1911, effective_date.month, effective_date.day
            ),
            "SecuritiesCompanyCode": "2330",
            "Name": "TSMC",
            "SuspensionOfTrading": "",
            "Reason": "",
        },
    ], ensure_ascii=False).encode("utf-8")


class TpexLifecycleCacheTests(unittest.TestCase):
    def test_cached_current_payload_with_later_effective_date_fails_closed(self):
        target = datetime.date(2026, 8, 3)
        effective = datetime.date(2026, 8, 4)
        payload = _tpex_payload(effective)

        with tempfile.TemporaryDirectory() as root:
            r = Path(root)
            entry = store_cached_raw_source(
                r,
                source_id="tpex_current_mode",
                target_date=target,
                payload=payload,
                parser_version=LIFECYCLE_PARSER_VERSION,
                source_url="https://example.com",
                fetched_at=datetime.datetime(2026, 8, 4, tzinfo=datetime.timezone.utc),
                date_verification="lifecycle_contract",
                effective_date=effective,
            )
            cached = load_cached_raw_source(
                r,
                source_id="tpex_current_mode",
                target_date=target,
                parser_version=LIFECYCLE_PARSER_VERSION,
            )
            self.assertIsNotNone(cached)
            self.assertEqual(cached.effective_date, effective)
            if cached is None:
                return
            payload_obj = json.loads(cached.payload.decode("utf-8-sig"))
            with self.assertRaises(HistoricalLifecycleUnavailable):
                _tpex_mode_events(payload_obj, entry.payload_sha256, target)

    def test_cache_effective_date_check_on_reload(self):
        target = datetime.date(2026, 8, 3)
        effective = datetime.date(2026, 8, 4)
        with tempfile.TemporaryDirectory() as root:
            r = Path(root)
            store_cached_raw_source(
                r,
                source_id="tpex_current_mode",
                target_date=target,
                payload=_tpex_payload(effective),
                parser_version=LIFECYCLE_PARSER_VERSION,
                source_url="https://example.com",
                fetched_at=datetime.datetime(2026, 8, 4, tzinfo=datetime.timezone.utc),
                date_verification="lifecycle_contract",
                effective_date=effective,
            )
            cached = load_cached_raw_source(
                r,
                source_id="tpex_current_mode",
                target_date=target,
                parser_version=LIFECYCLE_PARSER_VERSION,
            )
            self.assertIsNotNone(cached)
            if cached is None:
                return
            self.assertEqual(cached.effective_date, effective)
            self.assertNotEqual(cached.effective_date, target)

    def test_current_payload_accepted_when_effective_date_equals_target(self):
        target = datetime.date(2026, 8, 3)
        with tempfile.TemporaryDirectory() as root:
            r = Path(root)
            entry = store_cached_raw_source(
                r,
                source_id="tpex_current_mode",
                target_date=target,
                payload=_tpex_payload(target),
                parser_version=LIFECYCLE_PARSER_VERSION,
                source_url="https://example.com",
                fetched_at=datetime.datetime(2026, 8, 3, tzinfo=datetime.timezone.utc),
                date_verification="lifecycle_contract",
                effective_date=target,
            )
            cached = load_cached_raw_source(
                r,
                source_id="tpex_current_mode",
                target_date=target,
                parser_version=LIFECYCLE_PARSER_VERSION,
            )
            self.assertIsNotNone(cached)
            if cached is None:
                return
            self.assertEqual(cached.effective_date, target)
            payload_obj = json.loads(cached.payload.decode("utf-8-sig"))
            events = _tpex_mode_events(payload_obj, entry.payload_sha256, target)
            self.assertEqual(len(events), 0)

    def test_payload_sha_verification_is_cache_layer_responsibility(self):
        target = datetime.date(2026, 8, 3)
        with tempfile.TemporaryDirectory() as root:
            r = Path(root)
            store_cached_raw_source(
                r,
                source_id="tpex_current_mode",
                target_date=target,
                payload=_tpex_payload(target),
                parser_version=LIFECYCLE_PARSER_VERSION,
                source_url="https://example.com",
                fetched_at=datetime.datetime(2026, 8, 3, tzinfo=datetime.timezone.utc),
            )
            cached = load_cached_raw_source(
                r,
                source_id="tpex_current_mode",
                target_date=target,
                parser_version=LIFECYCLE_PARSER_VERSION,
            )
            self.assertIsNotNone(cached)
            if cached is None:
                return
            self.assertTrue(
                hashlib.sha256(cached.payload).hexdigest() == cached.payload_sha256
            )

    def test_extract_effective_date_from_current_mode_payload(self):
        effective = datetime.date(2026, 8, 4)
        payload = _tpex_payload(effective)
        extracted = _extract_current_mode_effective_date("tpex_current_mode", payload)
        self.assertEqual(extracted, effective)

    def test_extract_effective_date_from_non_current_source_returns_none(self):
        extracted = _extract_current_mode_effective_date("twse_current_stop", b"{}")
        self.assertIsNone(extracted)

    def test_directory_date_mismatch_with_payload_date_fails_closed(self):
        target = datetime.date(2026, 8, 3)
        wrong_date = datetime.date(2026, 8, 5)
        with tempfile.TemporaryDirectory() as root:
            r = Path(root)
            store_cached_raw_source(
                r,
                source_id="tpex_current_mode",
                target_date=target,
                payload=_tpex_payload(wrong_date),
                parser_version=LIFECYCLE_PARSER_VERSION,
                source_url="https://example.com",
                fetched_at=datetime.datetime(2026, 8, 5, tzinfo=datetime.timezone.utc),
                date_verification="lifecycle_contract",
                effective_date=wrong_date,
            )
            cached = load_cached_raw_source(
                r,
                source_id="tpex_current_mode",
                target_date=target,
                parser_version=LIFECYCLE_PARSER_VERSION,
            )
            self.assertIsNotNone(cached)
            if cached is None:
                return
            self.assertNotEqual(cached.effective_date, target)
            payload_obj = json.loads(cached.payload.decode("utf-8-sig"))
            with self.assertRaises(HistoricalLifecycleUnavailable):
                _tpex_mode_events(payload_obj, cached.payload_sha256, target)

    def test_later_fetch_for_different_target_preserves_earlier_cache(self):
        target1 = datetime.date(2026, 8, 3)
        target2 = datetime.date(2026, 8, 4)
        with tempfile.TemporaryDirectory() as root:
            r = Path(root)
            e1 = store_cached_raw_source(
                r,
                source_id="tpex_current_mode",
                target_date=target1,
                payload=_tpex_payload(target1),
                parser_version=LIFECYCLE_PARSER_VERSION,
                source_url="https://example.com",
                fetched_at=datetime.datetime(2026, 8, 3, tzinfo=datetime.timezone.utc),
                date_verification="lifecycle_contract",
                effective_date=target1,
            )
            e2 = store_cached_raw_source(
                r,
                source_id="tpex_current_mode",
                target_date=target2,
                payload=_tpex_payload(target2),
                parser_version=LIFECYCLE_PARSER_VERSION,
                source_url="https://example.com",
                fetched_at=datetime.datetime(2026, 8, 4, tzinfo=datetime.timezone.utc),
                date_verification="lifecycle_contract",
                effective_date=target2,
            )
            c1 = load_cached_raw_source(
                r,
                source_id="tpex_current_mode",
                target_date=target1,
                parser_version=LIFECYCLE_PARSER_VERSION,
            )
            c2 = load_cached_raw_source(
                r,
                source_id="tpex_current_mode",
                target_date=target2,
                parser_version=LIFECYCLE_PARSER_VERSION,
            )
            self.assertIsNotNone(c1)
            self.assertIsNotNone(c2)
            if c1 is None or c2 is None:
                return
            self.assertEqual(c1.effective_date, target1)
            self.assertEqual(c2.effective_date, target2)

    def test_historical_lifecycle_unavailable_is_a_value_error_subclass(self):
        self.assertTrue(issubclass(HistoricalLifecycleUnavailable, ValueError))


if __name__ == "__main__":
    unittest.main()
