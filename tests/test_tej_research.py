from __future__ import annotations

import datetime
import io
import json
import logging
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from stock_papi.quant.constants import MODEL_FEATURES
from stock_papi.research.tej import (
    TejClient,
    TejError,
    TejSchemaError,
    build_factor_snapshot,
    compare_official_truth,
    load_tej_raw_cache,
    normalize_pit_records,
    pit_asof_join,
    validate_factor_snapshot,
    write_tej_raw_cache,
)
from stock_papi.research.tej_challenger import (
    BASELINE_FEATURES,
    build_tej_challenger_frame,
    build_tej_challenger_spec,
    run_tej_challenger,
)
from stock_papi.research.evaluation import build_split_plan


class FakeResponseError(RuntimeError):
    def __init__(self, status_code):
        self.status_code = status_code
        super().__init__(f"fake response status={status_code}")


class FakeApiConfig:
    response = {"tables": [{"code": "TWN/DEMO"}]}
    error = None

    @classmethod
    def info(cls):
        if cls.error:
            raise cls.error
        return cls.response


class FakeTejApi:
    ApiConfig = FakeApiConfig
    get_calls = 0
    get_error_count = 0
    records = [{"coid": "2330", "mdate": "2025-01-02", "close": 100.0}]

    @classmethod
    def get(cls, table, **kwargs):
        cls.get_calls += 1
        if cls.get_error_count:
            cls.get_error_count -= 1
            raise FakeResponseError(429)
        if table != "TWN/DEMO":
            raise FakeResponseError(403)
        return list(cls.records)


class FakeClassifier:
    def fit(self, values, target):
        self.prevalence = float(sum(target) / len(target))
        return self

    def predict_proba(self, values):
        probabilities = [self.prevalence] * len(values)
        return np.column_stack((1.0 - np.array(probabilities), probabilities))


def _field_map():
    return {
        "entity": "coid",
        "effective_date": "fiscal_date",
        "available_at": "available_at",
        "fields": {
            "pe": "pe_value",
            "roe": "roe_value",
            "revenue_yoy": "revenue_yoy_value",
        },
    }


def _tej_rows():
    return [
        {
            "coid": "2330",
            "fiscal_date": "2024-12-31",
            "available_at": "2025-01-10T08:00:00Z",
            "pe_value": 18.0,
            "roe_value": 0.20,
            "revenue_yoy_value": 0.10,
        },
        {
            "coid": "2330",
            "fiscal_date": "2024-12-31",
            "available_at": "2025-02-10T08:00:00Z",
            "pe_value": 20.0,
            "roe_value": 0.25,
            "revenue_yoy_value": 0.15,
        },
        {
            "coid": "2317",
            "fiscal_date": "2024-12-31",
            "available_at": "2025-01-12T08:00:00Z",
            "pe_value": 12.0,
            "roe_value": 0.08,
            "revenue_yoy_value": 0.02,
        },
    ]


def _factor_document(rows):
    prepared = []
    for row in rows:
        item = dict(row)
        item.setdefault("source_payload_sha256", "a" * 64)
        item.setdefault("values", {})
        prepared.append(item)
    features = sorted(
        {
            feature
            for row in prepared
            for feature in (row.get("factors") or {})
        }
    )
    families = sorted(
        {
            feature.split("_", 2)[1].upper()
            for feature in features
        }
    )
    return {
        "schema_version": 1,
        "kind": "absorb-tej-factor-snapshot",
        "status": "available",
        "reason": None,
        "as_of": prepared[0]["factor_as_of"],
        "rows": prepared,
        "feature_manifest": features,
        "factor_families": families,
        "source_normalized_sha256": "b" * 64,
        "field_map_sha256": "c" * 64,
        "entity_map_sha256": "d" * 64,
        "production_model_changed": False,
    }


class TejResearchTests(unittest.TestCase):
    def setUp(self):
        FakeApiConfig.response = {"tables": [{"code": "TWN/DEMO"}]}
        FakeApiConfig.error = None
        FakeTejApi.get_calls = 0
        FakeTejApi.get_error_count = 0

    def test_disabled_by_default_does_not_call_api(self):
        with patch.dict(os.environ, {}, clear=True):
            client = TejClient.from_env(api=FakeTejApi)
            result = client.discover()
        self.assertEqual(result["status"], "disabled")
        self.assertEqual(FakeTejApi.get_calls, 0)

    def test_enabled_without_key_is_machine_readable_and_safe(self):
        with patch.dict(os.environ, {"TEJ_ENABLED": "true"}, clear=True):
            result = TejClient.from_env(api=FakeTejApi).discover()
        self.assertEqual(result["status"], "authentication_unavailable")
        self.assertNotIn("api_key", json.dumps(result))

    def test_authentication_failure_does_not_log_secret(self):
        FakeApiConfig.error = FakeResponseError(401)
        logger = logging.getLogger("tej-test")
        stream = io.StringIO()
        handler = logging.StreamHandler(stream)
        logger.addHandler(handler)
        try:
            client = TejClient(
                enabled=True,
                api_key="super-secret-api-key",
                api=FakeTejApi,
                logger=logger,
            )
            result = client.discover()
        finally:
            logger.removeHandler(handler)
        self.assertEqual(result["status"], "authentication_failed")
        self.assertNotIn("super-secret-api-key", stream.getvalue())
        self.assertNotIn("super-secret-api-key", json.dumps(result))

    def test_entitlement_discovery_distinguishes_entitled_and_missing_table(self):
        client = TejClient(
            enabled=True,
            api_key="secret",
            api=FakeTejApi,
        )
        self.assertEqual(client.check_dataset("TWN/DEMO")["status"], "dataset_entitled")
        self.assertEqual(client.check_dataset("TWN/NOT_ENTITLED")["status"], "dataset_not_entitled")

    def test_rate_limit_retries_are_bounded(self):
        FakeTejApi.get_error_count = 2
        sleeps = []
        client = TejClient(
            enabled=True,
            api_key="secret",
            api=FakeTejApi,
            sleep_fn=sleeps.append,
            max_retries=2,
        )
        result = client.fetch_dataset("TWN/DEMO")
        self.assertEqual(result["status"], "dataset_entitled")
        self.assertEqual(FakeTejApi.get_calls, 3)
        self.assertEqual(sleeps, [1, 2])

    def test_raw_cache_is_content_addressed_private_and_stable(self):
        metadata = {
            "provider": "TEJ",
            "dataset": "TWN/DEMO",
            "query": {"coid": ["2330"]},
            "requested_at": "2025-01-03T00:00:00Z",
            "retrieved_at": "2025-01-03T00:00:01Z",
            "effective_date": "2024-12-31",
            "available_at": "2025-01-02T08:00:00Z",
            "entity_field": "coid",
            "row_count": 1,
            "schema_version": 1,
            "client_version": "test",
        }
        payload = [{"coid": "2330", "value": 1.0}]
        with tempfile.TemporaryDirectory() as temporary:
            first = write_tej_raw_cache(Path(temporary), payload, metadata)
            second = write_tej_raw_cache(Path(temporary), payload, metadata)
            self.assertEqual(first, second)
            self.assertIn("raw", first["raw_path"])
            self.assertNotIn("publish", first["raw_path"])
            self.assertNotIn("api_key", Path(first["metadata_path"]).read_text())
            loaded = load_tej_raw_cache(Path(temporary), first["metadata_path"])
            self.assertEqual(loaded["payload"], payload)
            self.assertEqual(loaded["metadata"]["payload_sha256"], first["payload_sha256"])

            Path(first["raw_path"]).write_text("corrupt", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "immutable"):
                write_tej_raw_cache(Path(temporary), payload, metadata)

            nested = dict(metadata)
            nested["query"] = {"clauses": [{"api_key": "secret"}]}
            with self.assertRaisesRegex(ValueError, "credential"):
                write_tej_raw_cache(Path(temporary), payload, nested)

    def test_normalization_requires_explicit_available_at_and_mapping(self):
        with self.assertRaisesRegex(TejError, "available_at"):
            normalize_pit_records(
                [{"coid": "2330", "fiscal_date": "2024-12-31"}],
                table="TWN/DEMO",
                payload_sha256="a" * 64,
                field_map=_field_map(),
                entity_map={"2330": "2330"},
            )
        with self.assertRaisesRegex(TejError, "entity mapping"):
            normalize_pit_records(
                _tej_rows(),
                table="TWN/DEMO",
                payload_sha256="a" * 64,
                field_map=_field_map(),
                entity_map={"2330": "2330"},
            )

    def test_pit_asof_join_excludes_future_and_later_restatement(self):
        normalized = normalize_pit_records(
            _tej_rows(),
            table="TWN/DEMO",
            payload_sha256="a" * 64,
            field_map=_field_map(),
            entity_map={"2330": "2330", "2317": "2317"},
        )
        early = pit_asof_join(
            normalized,
            prediction_time="2025-01-31T00:00:00Z",
            effective_date="2024-12-31",
        )
        late = pit_asof_join(
            normalized,
            prediction_time="2025-02-20T00:00:00Z",
            effective_date="2024-12-31",
        )
        early_2330 = next(row for row in early if row["symbol"] == "2330")
        late_2330 = next(row for row in late if row["symbol"] == "2330")
        self.assertEqual(early_2330["values"]["pe"], 18.0)
        self.assertEqual(late_2330["values"]["pe"], 20.0)
        self.assertEqual(len(early), 2)
        self.assertEqual(
            pit_asof_join(
                normalized,
                prediction_time="2025-01-05T00:00:00Z",
                effective_date="2024-12-31",
            ),
            [],
        )
        conflicting = normalized + [
            dict(normalized[0], values={"pe": 999.0})
        ]
        with self.assertRaisesRegex(TejSchemaError, "same revision identity"):
            pit_asof_join(
                conflicting,
                prediction_time="2025-01-31T00:00:00Z",
                effective_date="2024-12-31",
            )

    def test_factor_snapshot_uses_only_visible_cross_section_and_declared_fields(self):
        normalized = normalize_pit_records(
            _tej_rows(),
            table="TWN/DEMO",
            payload_sha256="a" * 64,
            field_map=_field_map(),
            entity_map={"2330": "2330", "2317": "2317"},
        )
        snapshot = build_factor_snapshot(
            normalized,
            as_of="2025-01-31T00:00:00Z",
            effective_date="2024-12-31",
        )
        self.assertEqual(snapshot["status"], "available")
        self.assertIn("VALUE", snapshot["factor_families"])
        rows = {row["symbol"]: row for row in snapshot["rows"]}
        self.assertEqual(rows["2330"]["values"]["pe"], 18.0)
        self.assertNotIn("gross_margin", rows["2330"]["values"])
        self.assertIn("tej_value_pe_percentile", rows["2330"]["factors"])
        self.assertLessEqual(rows["2330"]["available_at"], "2025-01-31T00:00:00Z")
        validate_factor_snapshot(snapshot)

    def test_baseline_model_features_are_unchanged_and_challenger_is_separate(self):
        self.assertEqual(
            tuple(MODEL_FEATURES),
            (
                "MA_5", "MA20", "RET_1", "RET_5", "RET_20", "RSI", "Volat",
                "RANGE_PCT", "VOL_RATIO", "VOL_CHG", "INST_NET_RATIO", "MARGIN_CHG",
                "SHORT_CHG", "MACD_OSC", "K", "D", "MARKET_RET_1", "MARKET_RET_5",
                "MARKET_RET_20", "MARKET_VOL_20", "ETF50_RET_5", "STOCK_VS_MARKET_5", "OPTION_IV_LEVEL",
                "OPTION_IV_CHG_1", "OPTION_IV_CHG_5", "OPTION_IV_TERM_9D_3M",
                "OPTION_DATA_MISSING", "DATA_PRICE_DIFF_PCT", "DATA_PRICE_WARNING",
            ),
        )
        self.assertEqual(
            tuple(BASELINE_FEATURES),
            ("return_1", "momentum_5", "momentum_20", "volatility_20", "volume_ratio_20"),
        )
        self.assertNotIn("tej_value_pe_percentile", MODEL_FEATURES)
        frame = pd.DataFrame(
            [
                {
                    "symbol": "2330",
                    "source_market_date": "2025-01-31",
                    **{feature: 0.1 for feature in BASELINE_FEATURES},
                }
            ]
        )
        factors = [
            {
                "symbol": "2330",
                "effective_date": "2024-12-31",
                "available_at": "2025-01-10T00:00:00Z",
                "factor_as_of": "2025-01-10T00:00:00Z",
                "factors": {"tej_value_pe_percentile": 0.4},
            }
        ]
        factor_document = _factor_document(factors)
        result = build_tej_challenger_frame(frame, factor_document)
        self.assertIn("tej_value_pe_percentile", result["features"])
        self.assertIn("tej_value_pe_percentile", result["frame"].columns)
        self.assertEqual(result["eligible_count"], 1)
        future = _factor_document(
            [dict(factors[0], factor_as_of="2025-02-01T00:00:00Z")]
        )
        self.assertEqual(build_tej_challenger_frame(frame, future)["eligible_count"], 0)
        spec = build_tej_challenger_spec(result["features"])
        self.assertEqual(spec["model_version"], "tej-challenger-lgbm-v1")
        self.assertFalse(spec["production_eligible"])

    def test_shadow_mismatch_is_advisory_and_preserves_official_values(self):
        result = compare_official_truth(
            official_rows=[
                {"symbol": "2330", "market_date": "2025-01-02", "close": 100.0, "volume": 10}
            ],
            tej_rows=[
                {"symbol": "2330", "market_date": "2025-01-02", "close": 101.0, "volume": 10}
            ],
            official_identity={"source": "TWSE", "sha256": "b" * 64},
            tej_identity={"source": "TEJ", "sha256": "a" * 64},
            checked_at="2025-01-03T00:00:00Z",
        )
        self.assertEqual(result["status"], "mismatch")
        self.assertEqual(result["mismatches"][0]["official_value"], 100.0)
        self.assertEqual(result["mismatches"][0]["tej_value"], 101.0)
        self.assertFalse(result["override_official"])

        unavailable = compare_official_truth(
            official_rows=[{"symbol": "2330", "market_date": "2025-01-02", "close": 100.0}],
            tej_rows=[{"symbol": "2317", "market_date": "2025-01-02", "close": 100.0}],
            official_identity={"source": "TWSE", "sha256": "b" * 64},
            tej_identity={"source": "TEJ", "sha256": "a" * 64},
            checked_at="2025-01-03T00:00:00Z",
        )
        self.assertEqual(unavailable["status"], "unavailable")
        self.assertEqual(unavailable["reason"], "no_comparable_rows")

    def test_challenger_runs_on_same_pit_eligible_rows_without_promotion(self):
        rows = []
        for day in range(100):
            date = (datetime.date(2024, 1, 1) + datetime.timedelta(days=day)).isoformat()
            for symbol_index, symbol in enumerate(("2330", "2317")):
                rows.append(
                    {
                        "symbol": symbol,
                        "source_market_date": date,
                        "close": 100.0 + day + symbol_index,
                        "volume": 1000.0 + day,
                        **{feature: 0.1 + symbol_index for feature in BASELINE_FEATURES},
                        "future_return_5": 0.01 if (day + symbol_index) % 2 else -0.01,
                        "direction_5": int((day + symbol_index) % 2 == 1),
                    }
                )
        price_frame = pd.DataFrame(rows)
        factor_rows = [
            {
                "symbol": symbol,
                "effective_date": "2023-12-31",
                "available_at": "2023-12-31T00:00:00Z",
                "factor_as_of": "2023-12-31T00:00:00Z",
                "source_payload_sha256": "a" * 64,
                "factors": {"tej_value_pe_percentile": 0.4 + index * 0.1},
            }
            for index, symbol in enumerate(("2330", "2317"))
        ]
        joined = build_tej_challenger_frame(
            price_frame,
            _factor_document(factor_rows),
        )
        plan = build_split_plan(price_frame["source_market_date"].unique())
        result = run_tej_challenger(
            joined,
            plan,
            model_factory=FakeClassifier,
            bootstrap_iterations=50,
        )
        self.assertEqual(result["status"], "RUN")
        self.assertEqual(result["eligible_row_count"], len(price_frame))
        self.assertFalse(result["promotion"]["production_eligible"])

    def test_tej_scripts_are_separate_from_tw_writer(self):
        script = Path("scripts/run_tej_research.ps1").read_text(encoding="utf-8")
        self.assertIn("TEJ_ENABLED", script)
        self.assertNotIn("ABSORB-TW-PostClose", script)
        self.assertNotIn("run_tw_post_close_pipeline", script)


if __name__ == "__main__":
    unittest.main()
