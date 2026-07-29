import json
import tempfile
import unittest
from pathlib import Path

from tests.test_tw_official_historical import Response, Session, TARGET, payloads
from stock_papi.integrations.market_data.tw_official_bulk import OfficialSourceFailure
from stock_papi.integrations.market_data.tw_official_historical import (
    SOURCE_SCHEMA_VERSION,
    build_historical_daily_snapshot,
    parse_tpex_margin_report,
    parse_twse_margin_report,
)


class MutatingInstitutionalSession(Session):
    def get(self, url, *, params, **kwargs):
        response = super().get(url, params=params, **kwargs)
        source_id = self.calls[-1]["source_id"]
        if source_id != "tpex_institutional":
            return response
        document = json.loads(response.content.decode("utf-8"))
        document["tables"][0]["data"][0][0] = "9999"
        return Response(document)


class OfficialHardeningTests(unittest.TestCase):
    def test_margin_reports_require_exact_field_fingerprints(self):
        data = payloads(TARGET)
        fields = data["twse_margin"]["tables"][1]["fields"]
        fields[5], fields[6] = fields[6], fields[5]
        with self.assertRaisesRegex(ValueError, "target table"):
            parse_twse_margin_report(data["twse_margin"], TARGET)

        data = payloads(TARGET)
        fields = data["tpex_margin"]["tables"][0]["fields"]
        fields[6], fields[14] = fields[14], fields[6]
        with self.assertRaisesRegex(ValueError, "target table"):
            parse_tpex_margin_report(data["tpex_margin"], TARGET)

    def test_cache_hit_coverage_failure_is_structured(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first = build_historical_daily_snapshot(
                root,
                TARGET,
                session=Session(),
                minimum_price_symbols={"TWSE": 2, "TPEx": 2},
                minimum_chip_symbols=1,
            )
            self.assertEqual(first.source_schema_version, SOURCE_SCHEMA_VERSION)

            warm = Session()
            with self.assertRaises(OfficialSourceFailure) as caught:
                build_historical_daily_snapshot(
                    root,
                    TARGET,
                    session=warm,
                    minimum_price_symbols={"TWSE": 2, "TPEx": 2},
                )
            self.assertEqual(warm.calls, [])
            self.assertEqual(caught.exception.source_id, "twse_institutional")
            self.assertEqual(caught.exception.category, "schema_validation")

    def test_chip_sources_must_overlap_same_market_prices(self):
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaises(OfficialSourceFailure) as caught:
                build_historical_daily_snapshot(
                    Path(temporary),
                    TARGET,
                    session=MutatingInstitutionalSession(),
                    minimum_price_symbols={"TWSE": 2, "TPEx": 2},
                    minimum_chip_symbols=1,
                )
        self.assertEqual(caught.exception.source_id, "tpex_institutional")
        self.assertEqual(caught.exception.category, "cross_source_identity")

    def test_hardened_source_contract_uses_status_aware_v3_identity(self):
        self.assertEqual(SOURCE_SCHEMA_VERSION, "tw-official-historical-v3")


if __name__ == "__main__":
    unittest.main()
