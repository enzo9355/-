import ast
import datetime
import gzip
import inspect
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd

import local_quant
from stock_papi.quant.features import CALCULATED_COLUMNS, calc_all


class TWCalculatedColumnContractTests(unittest.TestCase):
    def test_calculated_columns_match_calc_all_assignments_in_order(self):
        from stock_papi.quant import features

        tree = ast.parse(inspect.getsource(features.calc_all))
        assigned = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.Assign):
                continue
            for target in node.targets:
                if (
                    isinstance(target, ast.Subscript)
                    and isinstance(target.value, ast.Name)
                    and target.value.id == "frame"
                    and isinstance(target.slice, ast.Constant)
                    and isinstance(target.slice.value, str)
                ):
                    assigned.append((node.lineno, target.slice.value))
        names = tuple(name for _, name in sorted(assigned))
        self.assertEqual(features.CALCULATED_COLUMNS, names)
        self.assertEqual(len(names), 20)


class TWHistoryPersistenceTests(unittest.TestCase):
    @staticmethod
    def _frame(days=25, *, end="2026-07-28"):
        index = pd.bdate_range(end=end, periods=days)
        values = [float(value) for value in range(100, 100 + len(index))]
        frame = pd.DataFrame(
            {
                "Open": values,
                "High": [value + 2 for value in values],
                "Low": [value - 1 for value in values],
                "Close": [value + 1 for value in values],
                "Volume": [1000 + value for value in values],
            },
            index=index,
        )
        frame.index.name = "Date"
        return frame

    @staticmethod
    def _pipeline(frame, *, run_ai_engine=None, run_latest_inference=None):
        return SimpleNamespace(
            get_data=lambda _symbol, _days: frame.copy(),
            calc_all=lambda data: calc_all(data, pd=pd, np=np),
            run_ai_engine=run_ai_engine or (lambda _data: {}),
            run_latest_inference=run_latest_inference,
            get_stock_name=lambda _symbol: "測試標的",
            PREDICTION_HORIZON=5,
        )

    @staticmethod
    def _artifact_daily(root, payload):
        path = local_quant.write_stock_artifact(root, "TW", "2330", payload)
        with gzip.open(path, "rt", encoding="utf-8") as stream:
            return json.load(stream)["daily"]

    @staticmethod
    def _daily_frame(rows):
        frame = pd.DataFrame(rows)
        frame["Date"] = pd.to_datetime(frame.pop("Date"))
        return frame.set_index("Date")

    @staticmethod
    def _daily_bytes(rows):
        return json.dumps(rows, separators=(",", ":"), ensure_ascii=False).encode()

    def test_canonical_frame_rejects_duplicate_dates_and_sorts_strictly(self):
        frame = self._frame(2)
        frame = frame.iloc[[1, 0]]

        canonical = local_quant._canonical_history_frame(frame)

        self.assertEqual(list(canonical.index), sorted(canonical.index))
        duplicate = frame.copy()
        duplicate.index = [frame.index[0], frame.index[0]]
        with self.assertRaisesRegex(ValueError, "historical market dates are duplicated"):
            local_quant._canonical_history_frame(duplicate)

    def test_warmup_rows_preserve_ohlcv_and_null_derived_fields(self):
        frame = self._frame()
        payload = local_quant.build_stock_snapshot(
            self._pipeline(frame), "TW", "2330"
        )

        self.assertEqual(len(payload["daily"]), len(frame))
        warmup = payload["daily"][0]
        self.assertEqual(warmup["Open"], frame.iloc[0]["Open"])
        self.assertEqual(warmup["High"], frame.iloc[0]["High"])
        self.assertEqual(warmup["Low"], frame.iloc[0]["Low"])
        self.assertEqual(warmup["Close"], frame.iloc[0]["Close"])
        self.assertEqual(warmup["Volume"], frame.iloc[0]["Volume"])
        self.assertTrue(all(warmup[name] is None for name in CALCULATED_COLUMNS))

    def test_latest_inference_ai_p_is_joined_after_mutation(self):
        frame = self._frame()

        def infer(data):
            data.loc[data.index[-1], "AI_P"] = 63.5
            return {"model_version": "lgbm-5d-v1"}

        payload = local_quant.build_stock_snapshot(
            self._pipeline(frame, run_latest_inference=infer),
            "TW",
            "2330",
            degraded_bootstrap=True,
        )

        self.assertEqual(len(payload["daily"]), len(frame))
        self.assertEqual(payload["latest"]["AI_P"], 63.5)
        self.assertIsNone(payload["daily"][0]["AI_P"])

    def test_oos_ai_p_is_joined_on_matching_dates(self):
        frame = self._frame()
        oos_date = frame.index[-2].date().isoformat()

        def run_ai_engine(data):
            data.loc[data.index[-2], "AI_P"] = 57.25
            return {"accuracy": 50.0}

        payload = local_quant.build_stock_snapshot(
            self._pipeline(frame, run_ai_engine=run_ai_engine), "TW", "2330"
        )
        by_date = {
            row["Date"].split("T", 1)[0]: row for row in payload["daily"]
        }

        self.assertEqual(len(payload["daily"]), len(frame))
        self.assertEqual(by_date[oos_date]["AI_P"], 57.25)
        self.assertIsNone(by_date[frame.index[0].date().isoformat()]["AI_P"])

    def test_retention_keeps_only_normal_730_day_request_result(self):
        source = self._frame(731, end="2026-07-28")
        calls = []

        def get_data(_symbol, days):
            calls.append(days)
            return source.tail(days).copy()

        pipeline = self._pipeline(source)
        pipeline.get_data = get_data
        payload = local_quant.build_stock_snapshot(pipeline, "TW", "2330")

        self.assertEqual(calls, [730])
        self.assertEqual(len(payload["daily"]), 730)
        self.assertEqual(
            payload["daily"][0]["Date"].split("T", 1)[0],
            source.index[-730].date().isoformat(),
        )

    def test_etf_history_preserves_warmup_rows(self):
        frame = self._frame()
        payload = local_quant.build_stock_snapshot(
            self._pipeline(frame), "TW", "0050", observation_only=True
        )

        self.assertEqual(len(payload["daily"]), len(frame))
        self.assertEqual(payload["daily"][0]["Close"], frame.iloc[0]["Close"])
        self.assertIsNone(payload["daily"][0]["MA20"])

    def test_short_history_still_fails_closed_when_latest_is_not_calculated(self):
        with self.assertRaisesRegex(ValueError, "calculated history is unavailable"):
            local_quant.build_stock_snapshot(
                self._pipeline(self._frame(19)), "TW", "2330"
            )

    def test_calculation_and_inference_mutation_cannot_overwrite_canonical_sources(self):
        frame = self._frame()
        frame["InstitutionalNet"] = list(range(len(frame)))
        frame["MARKET_RET_1"] = [index / 1000 for index in range(len(frame))]
        original = frame.copy(deep=True)
        observed = {}
        canonical_frames = []
        original_canonical = local_quant._canonical_history_frame

        def capture_canonical(data):
            canonical = original_canonical(data)
            canonical_frames.append(canonical)
            return canonical

        def mutate_and_calculate(data):
            observed["calculation_input"] = data
            data.loc[data.index[-1], [
                "Open", "High", "Low", "Close", "Volume",
                "InstitutionalNet", "MARKET_RET_1",
            ]] = [900.0, 902.0, 899.0, 901.0, 9999.0, -999.0, -0.9]
            return calc_all(data, pd=pd, np=np)

        def mutate_inference(data):
            observed["inference_input"] = data
            data.loc[data.index[-1], [
                "Open", "High", "Low", "Close", "Volume",
                "InstitutionalNet", "MARKET_RET_1",
            ]] = [800.0, 802.0, 799.0, 801.0, 8888.0, -888.0, -0.8]
            data.loc[data.index[-1], "AI_P"] = 63.5
            return {"model_version": "lgbm-5d-v1"}

        pipeline = self._pipeline(
            frame,
            run_latest_inference=mutate_inference,
        )
        pipeline.calc_all = mutate_and_calculate
        with patch.object(
            local_quant,
            "_canonical_history_frame",
            side_effect=capture_canonical,
        ):
            payload = local_quant.build_stock_snapshot(
                pipeline,
                "TW",
                "2330",
                degraded_bootstrap=True,
            )

        self.assertIsNot(observed["calculation_input"], frame)
        self.assertIsNot(observed["calculation_input"], canonical_frames[0])
        self.assertIsNot(observed["inference_input"], canonical_frames[0])
        allowed = set(CALCULATED_COLUMNS) | set(local_quant.OBSERVATION_MODEL_COLUMNS)
        by_date = {
            row["Date"].split("T", 1)[0]: row for row in payload["daily"]
        }
        for date, source in original.iterrows():
            persisted = by_date[date.date().isoformat()]
            for name in (
                "Open", "High", "Low", "Close", "Volume",
                "InstitutionalNet", "MARKET_RET_1",
            ):
                self.assertEqual(persisted[name], source[name])
            self.assertLessEqual(set(persisted) - {"Date"} - set(original), allowed)
        self.assertEqual(payload["latest"]["AI_P"], 63.5)

    def test_multistage_history_does_not_erode_and_rerun_is_byte_stable(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            local_quant.ensure_layout(root)
            current = self._frame(40, end="2026-07-28")
            stages = []
            first_rerun_bytes = None
            previous_canonical = None
            expected_bounds = (
                ("2026-06-03T00:00:00.000", "2026-07-28T00:00:00.000"),
                ("2026-06-03T00:00:00.000", "2026-07-29T00:00:00.000"),
                ("2026-06-03T00:00:00.000", "2026-07-30T00:00:00.000"),
                ("2026-06-03T00:00:00.000", "2026-07-31T00:00:00.000"),
                ("2026-06-03T00:00:00.000", "2026-07-31T00:00:00.000"),
            )
            targets = (None, *(datetime.date(2026, 7, day) for day in (29, 30, 31, 31)))
            for stage_index, target in enumerate(targets):
                if target is not None and target not in current.index.date:
                    previous = current.iloc[-1]
                    current.loc[pd.Timestamp(target)] = {
                        "Open": previous["Close"],
                        "High": previous["Close"] + 2,
                        "Low": previous["Close"] - 1,
                        "Close": previous["Close"] + 1,
                        "Volume": previous["Volume"] + 1,
                    }
                payload = local_quant.build_stock_snapshot(
                    self._pipeline(current),
                    "TW",
                    "2330",
                    target_market_date=target,
                    observation_only=True,
                )
                persisted = self._artifact_daily(root, payload)
                daily_bytes = self._daily_bytes(persisted)
                dates = [pd.Timestamp(row["Date"]) for row in persisted]
                canonical = [
                    {
                        name: row[name]
                        for name in ("Date", "Open", "High", "Low", "Close", "Volume")
                    }
                    for row in persisted
                ]
                self.assertEqual(
                    (persisted[0]["Date"], persisted[-1]["Date"]),
                    expected_bounds[stage_index],
                )
                self.assertTrue(all(left < right for left, right in zip(dates, dates[1:])))
                self.assertEqual(len(dates), len(set(dates)))
                if previous_canonical is not None:
                    self.assertEqual(
                        self._daily_bytes(canonical[:len(previous_canonical)]),
                        self._daily_bytes(previous_canonical),
                    )
                stages.append(
                    {
                        "rows": len(persisted),
                        "first": persisted[0]["Date"],
                        "last": persisted[-1]["Date"],
                        "unique": len({row["Date"] for row in persisted}),
                        "latest_calculated": all(
                            persisted[-1][name] is not None
                            for name in CALCULATED_COLUMNS
                        ),
                        "ohlcv": self._daily_bytes(canonical),
                    }
                )
                if target == datetime.date(2026, 7, 31):
                    if first_rerun_bytes is None:
                        first_rerun_bytes = daily_bytes
                    else:
                        self.assertEqual(daily_bytes, first_rerun_bytes)
                previous_canonical = canonical
                current = self._daily_frame(persisted)

        self.assertEqual([stage["rows"] for stage in stages], [40, 41, 42, 43, 43])
        self.assertTrue(all(stage["unique"] == stage["rows"] for stage in stages))
        self.assertTrue(all(stage["latest_calculated"] for stage in stages))
        self.assertEqual(stages[-1]["ohlcv"], stages[-2]["ohlcv"])
