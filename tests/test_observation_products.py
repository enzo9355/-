import copy
import datetime
import math
import unittest

from reporting.schemas import (
    LoadedReportSource,
    ReportSourceManifest,
    StockSnapshot,
)
from stock_papi.config.capabilities import PredictionCapabilityState
from stock_papi.batch.observation_products import build_observation_dashboard
from stock_papi.integrations.market_data.tw_trading_status import (
    evidence_sha256,
    resolve_lifecycle_status,
)


FORBIDDEN_KEYS = {
    "ai_p",
    "prob",
    "probability",
    "direction_score",
    "score",
    "recommendation",
    "top_picks",
    "model_version",
    "backtest_version",
}


def _stock(
    symbol,
    closes,
    *,
    name=None,
    ai_probability=99.0,
    rsi=55.0,
    volume_ratio=1.0,
    institution_ratio=0.0,
    foreign_net=0.0,
    sample_data=False,
):
    start = datetime.date(2026, 5, 13)
    rows = []
    for index, close in enumerate(closes):
        rows.append(
            {
                "Date": (start + datetime.timedelta(days=index)).isoformat()
                + "T00:00:00.000",
                "Close": float(close),
                "MA20": float(close) - 1.0,
                "MA60": float(close) - 2.0,
                "RSI": float(rsi),
                "VOL_RATIO": float(volume_ratio),
                "INST_NET_RATIO": float(institution_ratio),
                "ForeignNet": float(foreign_net),
                "DATA_PRICE_WARNING": 0.0,
                "OPTION_DATA_MISSING": 0.0,
                "AI_P": float(ai_probability),
            }
        )
    return StockSnapshot(
        symbol=symbol,
        name=name or symbol,
        market="TW",
        as_of=datetime.date.fromisoformat(rows[-1]["Date"][:10]),
        model_version="must-not-leak",
        daily=rows,
        backtest={"accuracy": 100.0},
        sha256="a" * 64,
        size=100,
        sample_data=sample_data,
    )


def _source(stocks, *, coverage=1.0, as_of=datetime.date(2026, 7, 16)):
    for stock in stocks:
        stock.as_of = as_of
        stock.daily[-1]["Date"] = as_of.isoformat() + "T00:00:00.000"
    return LoadedReportSource(
        manifest=ReportSourceManifest(
            schema_version=2,
            market="TW",
            generated_at="2026-07-16T10:00:00Z",
            market_as_of=as_of,
            universe_count=len(stocks),
            symbol_count=len(stocks),
            failure_count=0,
            failure_rate=0.0,
            coverage=coverage,
            failed_symbols=[],
            manifest_path="manifests/TW-20260716T100000Z-aaaaaaaaaaaa.json",
            manifest_sha256="a" * 64,
        ),
        stocks=stocks,
    )


def _no_trade_evidence(symbol, target):
    status = {
        "schema_version": 1,
        "status": "official_no_regular_trade",
        "market": "TW",
        "exchange": "TWSE",
        "symbol": symbol,
        "target_market_date": target.isoformat(),
        "source_id": "twse_price",
        "payload_sha256": "b" * 64,
        "raw_row_sha256": "c" * 64,
        "raw_fields": {
            "symbol": symbol,
            "name": f"測試股票 {symbol}",
            "open": "--",
            "high": "--",
            "low": "--",
            "close": "--",
            "volume": "0",
        },
        "parser_version": "tw-official-historical-parser-v3",
    }
    status["evidence_sha256"] = evidence_sha256(status)
    return status


def _suspended_evidence(symbol, target):
    event = {
        "schema_version": 1,
        "exchange": "TWSE",
        "symbol": symbol,
        "event_type": "suspend",
        "effective_date": (target - datetime.timedelta(days=1)).isoformat(),
        "source_id": "twse_reduction",
        "payload_sha256": "d" * 64,
        "raw_row_sha256": "e" * 64,
        "parser_version": "tw-lifecycle-parser-v2",
    }
    event["evidence_sha256"] = evidence_sha256(event)
    return resolve_lifecycle_status([event], target, active=True)


def _status_stock(symbol, target, status):
    stock = _stock(
        symbol,
        [1_000_000 + index for index in range(64)],
        name=f"測試股票 {symbol}",
        rsi=99,
        volume_ratio=999,
        institution_ratio=9,
        foreign_net=1_000_000,
    )
    stock.observation_as_of = target
    stock.latest_regular_price_date = stock.as_of
    stock.observation_kind = status["status"]
    stock.trading_status_evidence = status
    stock.sha256 = status["evidence_sha256"]
    return stock


def _source_v3(regular_stocks, status_stocks, target):
    for stock in regular_stocks:
        stock.as_of = target
        stock.observation_as_of = target
        stock.latest_regular_price_date = target
        stock.daily[-1]["Date"] = target.isoformat() + "T00:00:00.000"
    stocks = regular_stocks + status_stocks
    expected = {
        stock.symbol: {
            "status": stock.observation_kind,
            "evidence_sha256": stock.trading_status_evidence["evidence_sha256"],
            "artifact_sha256": stock.sha256,
            "latest_regular_price_date": stock.as_of.isoformat(),
        }
        for stock in status_stocks
    }
    return LoadedReportSource(
        manifest=ReportSourceManifest(
            schema_version=3,
            market="TW",
            generated_at="2026-07-16T10:00:00Z",
            market_as_of=target,
            universe_count=len(stocks),
            symbol_count=len(stocks),
            failure_count=0,
            failure_rate=0.0,
            coverage=1.0,
            failed_symbols=[],
            manifest_path="manifests/TW-20260716T100000Z-aaaaaaaaaaaa.json",
            manifest_sha256="a" * 64,
            target_market_date=target,
            observation_as_of=target,
            regular_price_symbol_count=len(regular_stocks),
            expected_non_price_symbol_count=len(status_stocks),
            operational_failure_count=0,
            regular_price_denominator=len(regular_stocks),
            regular_price_coverage=1.0,
            observation_coverage=1.0,
            expected_non_price_symbols=expected,
            operational_failed_symbols=[],
        ),
        stocks=stocks,
    )


def _capability():
    return PredictionCapabilityState(
        mode="research",
        observation_enabled=True,
        probability_allowed=False,
        ranking_allowed=False,
        strong_action_allowed=False,
        performance_endorsement_allowed=False,
        preview_candidate_prefix=None,
    )


def _walk_keys(value):
    if isinstance(value, dict):
        for key, item in value.items():
            yield str(key).lower()
            yield from _walk_keys(item)
    elif isinstance(value, list):
        for item in value:
            yield from _walk_keys(item)


class ObservationProductsTests(unittest.TestCase):
    def setUp(self):
        rising = [100 + index for index in range(65)]
        falling = [200 - index for index in range(65)]
        etf = [80 + index * 0.25 for index in range(65)]
        self.stocks = [
            _stock(
                "2330",
                rising,
                name="台積電",
                ai_probability=99,
                rsi=76,
                volume_ratio=2.4,
                institution_ratio=0.03,
                foreign_net=5000,
            ),
            _stock(
                "2317",
                falling,
                name="鴻海",
                ai_probability=1,
                rsi=24,
                volume_ratio=0.7,
                institution_ratio=-0.03,
                foreign_net=-4000,
            ),
            _stock("0050", etf, name="元大台灣50", ai_probability=88),
        ]
        self.industry_map = {
            "全市場": ["2330", "2317"],
            "半導體": ["2330"],
            "電子組裝": ["2317"],
            "ETF專區": ["0050"],
        }
        self.generated_at = datetime.datetime(
            2026, 7, 16, 10, 30, tzinfo=datetime.timezone.utc
        )

    def build(self, source=None, *, today=datetime.date(2026, 7, 17)):
        return build_observation_dashboard(
            source or _source(copy.deepcopy(self.stocks)),
            self.industry_map,
            _capability(),
            generated_at=self.generated_at,
            today=today,
        )

    def test_output_is_independent_of_ai_probability(self):
        high = _source(copy.deepcopy(self.stocks))
        low = copy.deepcopy(high)
        for stock in low.stocks:
            for row in stock.daily:
                row["AI_P"] = 1.0 if row["AI_P"] > 50 else 99.0

        self.assertEqual(self.build(high), self.build(low))

    def test_schema_contains_only_observation_domains(self):
        document = self.build()

        self.assertEqual(document["schema_version"], 2)
        self.assertEqual(document["kind"], "absorb-observation-dashboard")
        self.assertEqual(document["product_mode"], "observation")
        self.assertEqual(document["observation_as_of"], "2026-07-16")
        self.assertEqual(
            document["prediction_capability"]["mode"], "research"
        )
        self.assertFalse(
            document["prediction_capability"]["probability_allowed"]
        )
        self.assertTrue(document["market_observation"]["advancing_count"] > 0)
        self.assertTrue(document["market_observation"]["declining_count"] > 0)
        self.assertEqual(
            {item["name"] for item in document["industry_observations"]},
            {"半導體", "電子組裝"},
        )
        self.assertTrue(document["heatmap"])
        self.assertTrue(document["stock_events"])
        self.assertEqual(document["etf_observations"][0]["symbol"], "0050")
        self.assertTrue(document["daily_focus"])
        self.assertEqual(
            set(_walk_keys(document)).intersection(FORBIDDEN_KEYS), set()
        )

    def test_industry_display_order_uses_actual_relative_return(self):
        document = self.build()

        self.assertEqual(
            [item["name"] for item in document["industry_observations"]],
            ["半導體", "電子組裝"],
        )
        self.assertGreater(
            document["industry_observations"][0]["relative_return_5d_pct"],
            document["industry_observations"][1]["relative_return_5d_pct"],
        )
        self.assertEqual(
            document["heatmap"][0]["metric_name"], "relative_return_5d_pct"
        )

    def test_rejects_sample_low_coverage_stale_and_non_finite_sources(self):
        sample = _source(copy.deepcopy(self.stocks))
        sample.stocks[0].sample_data = True
        with self.assertRaisesRegex(ValueError, "sample"):
            self.build(sample)

        low_coverage = _source(copy.deepcopy(self.stocks), coverage=0.90)
        with self.assertRaisesRegex(ValueError, "coverage"):
            self.build(low_coverage)

        stale = _source(
            copy.deepcopy(self.stocks), as_of=datetime.date(2026, 7, 1)
        )
        with self.assertRaisesRegex(ValueError, "stale"):
            self.build(stale, today=datetime.date(2026, 7, 17))

        non_finite = _source(copy.deepcopy(self.stocks))
        non_finite.stocks[0].daily[-1]["Close"] = math.nan
        with self.assertRaisesRegex(ValueError, "finite"):
            self.build(non_finite)

    def test_source_freshness_accepts_day_7_and_rejects_day_8(self):
        source = _source(
            copy.deepcopy(self.stocks),
            as_of=datetime.date(2026, 7, 16),
        )

        accepted = self.build(
            copy.deepcopy(source),
            today=datetime.date(2026, 7, 23),
        )
        self.assertEqual(accepted["data_quality"]["source_age_days"], 7)

        with self.assertRaisesRegex(ValueError, "stale"):
            self.build(
                copy.deepcopy(source),
                today=datetime.date(2026, 7, 24),
            )

    def test_output_order_is_deterministic(self):
        first = self.build(_source(copy.deepcopy(self.stocks)))
        second = self.build(_source(list(reversed(copy.deepcopy(self.stocks)))))

        self.assertEqual(first, second)

    def test_v3_status_rows_are_excluded_from_price_math_and_rendered_separately(self):
        target = datetime.date(2026, 7, 16)
        no_trade = _status_stock(
            "2303", target, _no_trade_evidence("2303", target)
        )
        suspended = _status_stock(
            "1459", target, _suspended_evidence("1459", target)
        )
        industry_map = copy.deepcopy(self.industry_map)
        industry_map["全市場"].extend(["2303", "1459"])
        industry_map["半導體"].extend(["2303", "1459"])
        regular = _source(copy.deepcopy(self.stocks))
        source = _source_v3(
            copy.deepcopy(self.stocks), [no_trade, suspended], target
        )

        baseline = build_observation_dashboard(
            regular,
            self.industry_map,
            _capability(),
            generated_at=self.generated_at,
            today=datetime.date(2026, 7, 17),
        )
        document = build_observation_dashboard(
            source,
            industry_map,
            _capability(),
            generated_at=self.generated_at,
            today=datetime.date(2026, 7, 17),
        )

        for key in ("market_observation", "industry_observations", "heatmap", "stock_events", "etf_observations"):
            self.assertEqual(document[key], baseline[key])
        self.assertEqual(
            [item["label"] for item in document["trading_status_observations"]],
            ["停止買賣", "當日無正常交易"],
        )
        for item in document["trading_status_observations"]:
            self.assertEqual(item["observation_as_of"], target.isoformat())
            self.assertEqual(
                set(item),
                {
                    "symbol",
                    "name",
                    "status",
                    "label",
                    "observation_as_of",
                    "latest_regular_price_date",
                    "evidence_sha256",
                    "last_regular_close",
                },
            )
        self.assertEqual(document["data_quality"]["regular_price_count"], 3)
        self.assertEqual(document["data_quality"]["verified_status_count"], 2)
        self.assertEqual(document["data_quality"]["operational_failure_count"], 0)

    def test_v3_rejects_status_evidence_tampering(self):
        target = datetime.date(2026, 7, 16)
        status = _status_stock(
            "2303", target, _no_trade_evidence("2303", target)
        )
        status.trading_status_evidence["raw_fields"]["volume"] = "1"
        source = _source_v3(copy.deepcopy(self.stocks), [status], target)

        with self.assertRaisesRegex(ValueError, "status"):
            build_observation_dashboard(
                source,
                self.industry_map,
                _capability(),
                generated_at=self.generated_at,
                today=datetime.date(2026, 7, 17),
            )


if __name__ == "__main__":
    unittest.main()
