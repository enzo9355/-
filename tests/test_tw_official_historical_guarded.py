import copy
import json
import tempfile
import unittest
from pathlib import Path

from tests.test_tw_official_historical import (
    Response,
    Session,
    TARGET,
    payloads,
)

from stock_papi.integrations.market_data.tw_official_bulk import (
    OfficialSourceFailure,
)
from stock_papi.integrations.market_data.tw_official_historical_guarded import (
    build_historical_daily_snapshot,
    parse_tpex_margin_report,
    parse_twse_margin_report,
)


class MutatingSession(Session):
    def get(self, url, *, params, **kwargs):
        response = super().get(url, params=params, **kwargs)
        source_id = self.calls[-1][0]
        if source_id != "tpex_institutional":
            return response
        document = json.loads(response.content.decode("utf-8"))
        document["tables"][0]["data"][0][0] = "9999"
        return Response(document)


class GuardedSchemaTests(unittest.TestCase):
    def test_twse_margin_requires_the_complete_field_fingerprint(self):
        document = copy.deepcopy(payloads(TARGET)["twse_margin"])
        fields = document["tables"][1]["fields"]
        fields[6], fields[12] = fields[12], fields[6]
        with self.assertRaises(ValueError):
            parse_twse_margin_report(document, TARGET)

    def test_tpex_margin_requires_the_complete_field_fingerprint(self):
        document = copy.deepcopy(payloads(TARGET)["tpex_margin"])
        fields = document["tables"][0]["fields"]
        fields[6], fields[14] = fields[14], fields[6]
        with self.assertRaises(ValueError):
            parse_tpex_margin_report(document, TARGET)


class GuardedCoverageTests(unittest.TestCase):
    def test_default_core_coverage_rejects_a_one_symbol_report(self):
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaises(OfficialSourceFailure) as caught:
                build_historical_daily_snapshot(
                    Path(temporary),
                    TARGET,
                    session=Session(),
                    minimum_price_symbols={"TWSE": 2, "TPEx": 2},
                )
        self.assertEqual(caught.exception.category, "schema_validation")
        self.assertIn("coverage", caught.exception.safe_message)

    def test_fixture_sized_overrides_still_build_all_six_sources(self):
        with tempfile.TemporaryDirectory() as temporary:
            session = Session()
            snapshot = build_historical_daily_snapshot(
                Path(temporary),
                TARGET,
                session=session,
                minimum_price_symbols={"TWSE": 2, "TPEx": 2},
                minimum_core_symbols={"TWSE": 1, "TPEx": 1},
            )
        self.assertEqual(len(session.calls), 6)
        self.assertEqual(snapshot.request_count, 6)
        self.assertEqual(snapshot.source_schema_version, "tw-official-historical-v2")

    def test_chip_symbols_must_overlap_the_same_market_price_source(self):
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaises(OfficialSourceFailure) as caught:
                build_historical_daily_snapshot(
                    Path(temporary),
                    TARGET,
                    session=MutatingSession(),
                    minimum_price_symbols={"TWSE": 2, "TPEx": 2},
                    minimum_core_symbols={"TWSE": 1, "TPEx": 1},
                )
        self.assertEqual(caught.exception.category, "cross_source_identity")


if __name__ == "__main__":
    unittest.main()
