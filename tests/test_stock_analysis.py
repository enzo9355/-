import unittest

from stock_papi.services.stock_analysis import snapshot_dataframe
from tests.report_fixtures import warmup_stock_document


class StockAnalysisCompatibilityTests(unittest.TestCase):
    def test_snapshot_dataframe_filters_complete_features_before_history_gate(self):
        import pandas as pd

        frame = snapshot_dataframe(
            warmup_stock_document("2330", rows=220, warmup_rows=20), pd=pd
        )

        self.assertEqual(len(frame), 200)
        self.assertIsNone(snapshot_dataframe(
            warmup_stock_document("2330", rows=219, warmup_rows=20), pd=pd
        ))

    def test_snapshot_dataframe_rejects_bool_and_non_finite_required_values(self):
        import pandas as pd

        document = warmup_stock_document("2330", rows=203, warmup_rows=0)
        for value in (True, float("inf"), float("-inf")):
            document["daily"][-1]["RSI"] = value
            self.assertIsNone(snapshot_dataframe(document, pd=pd))

    def test_snapshot_dataframe_rejects_stale_complete_prefix_when_latest_is_not_ready(self):
        import pandas as pd

        for value in (None, True, float("inf")):
            document = warmup_stock_document("2330", rows=201, warmup_rows=0)
            document["daily"][-1]["AI_P"] = value
            self.assertIsNone(snapshot_dataframe(document, pd=pd))


if __name__ == "__main__":
    unittest.main()
