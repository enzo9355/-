import datetime
import hashlib
import json
import unittest

from stock_papi.repositories.prediction_snapshots import load_prediction_snapshot


def canonical(value):
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


class PredictionSnapshotRepositoryTests(unittest.TestCase):
    def product(self):
        return {
            "schema_version": 1,
            "kind": "absorb-five-session-predictions",
            "market": "TW",
            "as_of": "2026-08-26",
            "generated_at": "2026-08-26T14:00:00Z",
            "horizon_sessions": 5,
            "source_manifest": "quant/v1/manifests/TW-20260826T130000Z-aaaaaaaaaaaa.json",
            "source_manifest_sha256": "a" * 64,
            "backtest_sha256": "b" * 64,
            "model_version": "lgbm-5d-v1",
            "feature_schema_version": 1,
            "entities": {
                "2330": {
                    "symbol": "2330", "entity_type": "security",
                    "as_of": "2026-08-26", "target_session": "2026-09-02",
                    "current_price": 100.0, "up_probability": 0.645,
                    "predicted_return_5d": 0.04, "predicted_price": 104.0,
                    "predicted_change_pct": 4.0,
                }
            },
        }

    def payloads(self, *, pointer_changes=None, product_changes=None):
        product = self.product()
        product.update(product_changes or {})
        body = canonical(product)
        digest = hashlib.sha256(body).hexdigest()
        pointer = {
            "schema_version": 1,
            "kind": "absorb-five-session-predictions-pointer",
            "market": "TW",
            "as_of": "2026-08-26",
            "path": f"objects/{digest}.json",
            "sha256": digest,
            "size": len(body),
            "source_manifest": product["source_manifest"],
            "source_manifest_sha256": product["source_manifest_sha256"],
            "backtest_sha256": product["backtest_sha256"],
        }
        pointer.update(pointer_changes or {})
        return canonical(pointer), body

    def test_reader_verifies_pointer_hash_and_document(self):
        payloads = iter(self.payloads())
        result = load_prediction_snapshot(
            "TW", today=datetime.date(2026, 8, 27),
            load_object=lambda _path, _size: next(payloads), cache={},
        )

        self.assertEqual(result["entities"]["2330"]["predicted_price"], 104.0)

    def test_reader_rejects_hash_mismatch_and_stale_product(self):
        for payloads in (
            self.payloads(pointer_changes={"sha256": "0" * 64}),
            self.payloads(product_changes={"as_of": "2026-08-01"}),
        ):
            with self.subTest(payloads=payloads):
                values = iter(payloads)
                self.assertIsNone(load_prediction_snapshot(
                    "TW", today=datetime.date(2026, 8, 27),
                    load_object=lambda _path, _size: next(values), cache={},
                ))

    def test_reader_accepts_research_pointer_without_backtest_claim(self):
        product = self.product()
        product["schema_version"] = 2
        product["validation_mode"] = "research"
        product.pop("backtest_sha256")
        product.update({
            "source_symbol_count": 1,
            "prediction_count": 1,
            "unavailable_count": 0,
            "unavailable_symbols": [],
        })
        body = canonical(product)
        digest = hashlib.sha256(body).hexdigest()
        pointer = {
            "schema_version": 2,
            "kind": "absorb-five-session-predictions-pointer",
            "market": "TW",
            "as_of": "2026-08-26",
            "path": f"objects/{digest}.json",
            "sha256": digest,
            "size": len(body),
            "source_manifest": product["source_manifest"],
            "source_manifest_sha256": product["source_manifest_sha256"],
            "validation_mode": "research",
        }
        payloads = iter((canonical(pointer), body))

        result = load_prediction_snapshot(
            "TW", today=datetime.date(2026, 8, 27),
            load_object=lambda _path, _size: next(payloads), cache={},
        )

        self.assertEqual(result["validation_mode"], "research")


if __name__ == "__main__":
    unittest.main()
