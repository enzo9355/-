import datetime
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from stock_papi.batch.calendar import TradingCalendarSet
from stock_papi.batch.pre_market import (
    OVERNIGHT_SYMBOLS,
    PreMarketPipeline,
    PreMarketPipelineError,
)
from stock_papi.batch.runtime import job_namespace
from stock_papi.integrations.market_data.us_calendar import get_us_calendar_documents
from reporting.schemas import StockSnapshot


UTC = datetime.timezone.utc


class Calendar:
    def latest_session_on_or_before(self, value):
        return datetime.date(2026, 7, 14)

    def session_offset(self, value, offset):
        return datetime.date(2026, 7, 13)


def real_calendars():
    return TradingCalendarSet.from_documents(get_us_calendar_documents(2026, 2026))


def base_receipt():
    metadata = {
        "schema_version": 2,
        "kind": "absorb-report",
        "product_mode": "observation",
        "report_type": "post_close",
        "market": "TW",
        "source_market_date": "2026-07-14",
        "applicable_trading_date": "2026-07-15",
        "published_at": "2026-07-14T10:00:00Z",
        "forecast_start_date": "2026-07-15",
        "forecast_end_date": "2026-07-15",
        "observation_start_date": "2026-07-14",
        "observation_end_date": "2026-07-15",
        "backtest_as_of": None,
        "data_as_of": "2026-07-14",
        "source_manifest": "quant/v1/manifests/TW-20260714T090000Z-aaaaaaaaaaaa.json",
        "source_manifest_sha256": "a" * 64,
        "model_versions": {},
        "prediction_capability": {
            "mode": "research", "observation_enabled": True,
            "probability_allowed": False, "ranking_allowed": False,
            "strong_action_allowed": False,
            "performance_endorsement_allowed": False,
        },
        "title": "盤後市場觀察", "summary": [], "warnings": [],
        "content": {
            "market_observation": {
                "return_1d_pct": 1.0, "ma20_breadth_pct": 50.0,
                "realized_volatility_20d_pct": 10.0,
                "advancing_count": 10, "declining_count": 5,
            },
            "data_quality": {"coverage": 1.0, "symbol_count": 15, "failure_count": 0},
            "daily_focus": ["台股摘要"],
        },
    }
    encoded = json.dumps(metadata, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    return {"metadata": metadata, "metadata_sha256": hashlib.sha256(encoded).hexdigest()}


def source(symbols=OVERNIGHT_SYMBOLS):
    manifest = SimpleNamespace(
        market="US", market_as_of=datetime.date(2026, 7, 14),
        manifest_path="manifests/US-20260714T220000Z-aaaaaaaaaaaa.json",
        manifest_sha256="a" * 64,
    )
    stocks = []
    for index, symbol in enumerate(symbols):
        stocks.append(StockSnapshot(
            symbol=symbol, name=symbol, market="US", as_of=manifest.market_as_of,
            model_version="test", daily=[
                {"Date": "2026-07-13T00:00:00", "Close": 100.0},
                {"Date": "2026-07-14T00:00:00", "Close": 101.0 if index < 4 else 99.0},
            ], backtest={}, sha256="b" * 64, size=1,
        ))
    return SimpleNamespace(manifest=manifest, stocks=stocks)


def source_for(completed, previous, closes):
    manifest = SimpleNamespace(
        market="US", market_as_of=completed,
        manifest_path=f"manifests/US-{completed:%Y%m%d}T220000Z-aaaaaaaaaaaa.json",
        manifest_sha256="a" * 64,
    )
    stocks = []
    for symbol in OVERNIGHT_SYMBOLS:
        prev_close, current_close = closes[symbol]
        stocks.append(StockSnapshot(
            symbol=symbol, name=symbol, market="US", as_of=completed,
            model_version="test", daily=[
                {"Date": f"{previous}T00:00:00", "Close": prev_close},
                {"Date": f"{completed}T00:00:00", "Close": current_close},
            ], backtest={}, sha256="b" * 64, size=1,
        ))
    return SimpleNamespace(manifest=manifest, stocks=stocks)


class PreMarketQuantOverlayTests(unittest.TestCase):
    def test_uses_two_completed_us_sessions_and_five_fixed_symbols(self):
        published = []
        with tempfile.TemporaryDirectory() as root:
            result = PreMarketPipeline(
                Path(root), applicable_trading_date=datetime.date(2026, 7, 15),
                load_base=base_receipt, source_loaders=[],
                us_source_loader=source, us_calendars=Calendar(),
                publish=lambda value: published.append(value) or {},
                notify=lambda value: {},
            ).run(now=datetime.datetime(2026, 7, 15, 0, tzinfo=UTC))
        overlay = result["outputs"]["metadata"]["content"]["overnight_overlay"]
        self.assertEqual(overlay["status"], "risk_on")
        self.assertEqual(overlay["as_of"], "2026-07-14")
        self.assertEqual(overlay["previous_as_of"], "2026-07-13")
        self.assertEqual([item["symbol"] for item in overlay["symbols"]], list(OVERNIGHT_SYMBOLS))
        self.assertEqual(len(published), 1)

    def test_incomplete_universe_fails_closed_before_publish(self):
        published = []
        with tempfile.TemporaryDirectory() as root:
            with self.assertRaises(PreMarketPipelineError):
                PreMarketPipeline(
                    Path(root), applicable_trading_date=datetime.date(2026, 7, 15),
                    load_base=base_receipt, source_loaders=[],
                    us_source_loader=lambda: source(OVERNIGHT_SYMBOLS[:-1]),
                    us_calendars=Calendar(),
                    publish=lambda value: published.append(value), notify=lambda value: {},
                ).run(now=datetime.datetime(2026, 7, 15, 0, tzinfo=UTC))
        self.assertEqual(published, [])

    def test_superset_source_with_extra_symbol_still_outputs_fixed_five(self):
        published = []
        with tempfile.TemporaryDirectory() as root:
            result = PreMarketPipeline(
                Path(root), applicable_trading_date=datetime.date(2026, 7, 15),
                load_base=base_receipt, source_loaders=[],
                us_source_loader=lambda: source(OVERNIGHT_SYMBOLS + ("AAPL",)),
                us_calendars=Calendar(),
                publish=lambda value: published.append(value) or {},
                notify=lambda value: {},
            ).run(now=datetime.datetime(2026, 7, 15, 0, tzinfo=UTC))
        overlay = result["outputs"]["metadata"]["content"]["overnight_overlay"]
        self.assertEqual(len(overlay["symbols"]), len(OVERNIGHT_SYMBOLS))
        self.assertEqual(
            [item["symbol"] for item in overlay["symbols"]], list(OVERNIGHT_SYMBOLS)
        )
        self.assertEqual(len(published), 1)

    def test_weekend_and_holiday_resolve_to_last_completed_us_session(self):
        calendars = real_calendars()
        scenarios = [
            (
                datetime.datetime(2026, 7, 12, 20, tzinfo=UTC),
                datetime.date(2026, 7, 10),
                datetime.date(2026, 7, 9),
            ),
            (
                datetime.datetime(2026, 7, 4, 20, tzinfo=UTC),
                datetime.date(2026, 7, 2),
                datetime.date(2026, 7, 1),
            ),
        ]
        for now, completed, previous in scenarios:
            with self.subTest(now=now, completed=completed):
                published = []
                with tempfile.TemporaryDirectory() as root:
                    result = PreMarketPipeline(
                        Path(root), applicable_trading_date=datetime.date(2026, 7, 15),
                        load_base=base_receipt, source_loaders=[],
                        us_source_loader=lambda: source_for(
                            completed, previous,
                            {symbol: (100.0, 101.0) for symbol in OVERNIGHT_SYMBOLS},
                        ),
                        us_calendars=calendars,
                        publish=lambda value: published.append(value) or {},
                        notify=lambda value: {},
                    ).run(now=now)
                overlay = result["outputs"]["metadata"]["content"]["overnight_overlay"]
                self.assertEqual(overlay["as_of"], completed.isoformat())
                self.assertEqual(overlay["previous_as_of"], previous.isoformat())
                self.assertEqual(overlay["status"], "risk_on")
                self.assertEqual(len(published), 1)

    def test_stale_manifest_fails_closed_before_publish(self):
        published = []
        with tempfile.TemporaryDirectory() as root:
            with self.assertRaises(PreMarketPipelineError):
                PreMarketPipeline(
                    Path(root), applicable_trading_date=datetime.date(2026, 7, 15),
                    load_base=base_receipt, source_loaders=[],
                    us_source_loader=lambda: source_for(
                        datetime.date(2026, 7, 10), datetime.date(2026, 7, 9),
                        {symbol: (100.0, 101.0) for symbol in OVERNIGHT_SYMBOLS},
                    ),
                    us_calendars=Calendar(),
                    publish=lambda value: published.append(value), notify=lambda value: {},
                ).run(now=datetime.datetime(2026, 7, 15, 0, tzinfo=UTC))
        self.assertEqual(published, [])

    def test_zero_return_is_unchanged_and_status_is_neutral(self):
        published = []
        with tempfile.TemporaryDirectory() as root:
            result = PreMarketPipeline(
                Path(root), applicable_trading_date=datetime.date(2026, 7, 15),
                load_base=base_receipt, source_loaders=[],
                us_source_loader=lambda: source_for(
                    datetime.date(2026, 7, 14), datetime.date(2026, 7, 13),
                    {symbol: (100.0, 100.0) for symbol in OVERNIGHT_SYMBOLS},
                ),
                us_calendars=Calendar(),
                publish=lambda value: published.append(value) or {},
                notify=lambda value: {},
            ).run(now=datetime.datetime(2026, 7, 15, 0, tzinfo=UTC))
        overlay = result["outputs"]["metadata"]["content"]["overnight_overlay"]
        self.assertEqual(overlay["status"], "neutral")
        self.assertEqual(
            {item["direction"] for item in overlay["symbols"]},
            {"unchanged"},
        )
        self.assertTrue(
            all(item["return_pct"] == 0.0 for item in overlay["symbols"])
        )
        self.assertEqual(len(published), 1)

    def _checkpoint_path(self, root, base, applicable):
        identity = {
            "applicable_trading_date": applicable.isoformat(),
            "base_metadata_sha256": base["metadata_sha256"],
        }
        digest = hashlib.sha256(
            json.dumps(
                identity, ensure_ascii=False, sort_keys=True, separators=(",", ":")
            ).encode("utf-8")
        ).hexdigest()[:8]
        run_id = f"{applicable:%Y%m%d}T000000Z-{digest}"
        current = job_namespace(root, "pre_market_update").checkpoint
        return current.with_name(f"{run_id}.json")

    def test_old_checkpoint_contract_cannot_bypass_new_contract(self):
        base = base_receipt()
        applicable = datetime.date(2026, 7, 15)
        published = []
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            checkpoint = self._checkpoint_path(root, base, applicable)
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            checkpoint.write_text(json.dumps({
                "schema_version": 1,
                "job_type": "pre_market_update",
                "run_id": checkpoint.name[:-5],
                "applicable_trading_date": applicable.isoformat(),
                "base_metadata_sha256": base["metadata_sha256"],
                "completed_stages": ["metadata", "publish", "notify"],
                "outputs": {
                    "metadata": {
                        "content": {
                            "overnight_overlay": {
                                "status": "insufficient",
                                "message": "資料不足",
                                "as_of": "2026-07-14T23:30:00Z",
                                "available": [],
                                "unavailable": [],
                            },
                        },
                    },
                },
                "status": "completed",
            }), encoding="utf-8")
            with self.assertRaises(PreMarketPipelineError):
                PreMarketPipeline(
                    root, applicable_trading_date=applicable,
                    load_base=lambda: base, source_loaders=[],
                    us_source_loader=source, us_calendars=Calendar(),
                    publish=lambda value: published.append(value), notify=lambda value: {},
                ).run(now=datetime.datetime(2026, 7, 15, 0, tzinfo=UTC))
        self.assertEqual(published, [])

    def test_old_checkpoint_resume_cannot_bypass_new_contract(self):
        base = base_receipt()
        applicable = datetime.date(2026, 7, 15)
        published = []
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            checkpoint = self._checkpoint_path(root, base, applicable)
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            checkpoint.write_text(json.dumps({
                "schema_version": 1,
                "job_type": "pre_market_update",
                "run_id": checkpoint.name[:-5],
                "applicable_trading_date": applicable.isoformat(),
                "base_metadata_sha256": base["metadata_sha256"],
                "completed_stages": ["metadata"],
                "outputs": {
                    "metadata": {
                        "content": {
                            "overnight_overlay": {
                                "status": "insufficient",
                                "message": "資料不足",
                                "as_of": "2026-07-14T23:30:00Z",
                                "available": [],
                                "unavailable": [],
                            },
                        },
                    },
                },
                "status": "running",
            }), encoding="utf-8")
            with self.assertRaises(PreMarketPipelineError):
                PreMarketPipeline(
                    root, applicable_trading_date=applicable,
                    load_base=lambda: base, source_loaders=[],
                    us_source_loader=source, us_calendars=Calendar(),
                    publish=lambda value: published.append(value), notify=lambda value: {},
                ).run(now=datetime.datetime(2026, 7, 15, 0, tzinfo=UTC))
        self.assertEqual(published, [])


if __name__ == "__main__":
    unittest.main()
