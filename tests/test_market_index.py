import datetime
import unittest

from stock_papi.services.market_index import (
    fetch_twse_index_snapshot,
    parse_twse_index_month,
)


def _payload(month, rows):
    return {
        "stat": "OK",
        "date": month,
        "title": "發行量加權股價指數歷史資料",
        "fields": ["日期", "開盤指數", "最高指數", "最低指數", "收盤指數"],
        "data": rows,
    }


class _Response:
    status_code = 200

    def __init__(self, payload):
        self._payload = payload
        self.content = b"{}"

    def json(self):
        return self._payload


class MarketIndexTests(unittest.TestCase):
    def test_parser_accepts_exact_twse_index_schema_and_roc_dates(self):
        rows = parse_twse_index_month(
            _payload(
                "20260801",
                [["115/08/25", "44,728.36", "45,169.46", "44,210.31", "45,169.46"]],
            ),
            datetime.date(2026, 8, 1),
        )

        self.assertEqual(rows[0]["time"], "2026-08-25")
        self.assertEqual(rows[0]["close"], 45169.46)

    def test_parser_rejects_invalid_ohlc_relationship(self):
        with self.assertRaisesRegex(ValueError, "OHLC"):
            parse_twse_index_month(
                _payload(
                    "20260801",
                    [["115/08/25", "44,728.36", "44,000", "44,210.31", "45,169.46"]],
                ),
                datetime.date(2026, 8, 1),
            )

    def test_snapshot_is_bounded_to_verified_dashboard_date(self):
        calls = []

        def get(_url, *, params, headers, timeout):
            calls.append((params["date"], headers["User-Agent"], timeout))
            month = params["date"]
            year = int(month[:4])
            month_number = int(month[4:6])
            roc_year = year - 1911
            rows = []
            for day in range(1, 29):
                date = datetime.date(year, month_number, day)
                if date.weekday() >= 5:
                    continue
                close = 40000 + len(calls) * 100 + day
                rows.append([
                    f"{roc_year:03d}/{month_number:02d}/{day:02d}",
                    f"{close - 10:,.2f}",
                    f"{close + 20:,.2f}",
                    f"{close - 30:,.2f}",
                    f"{close:,.2f}",
                ])
            return _Response(_payload(month, rows))

        result = fetch_twse_index_snapshot(
            datetime.date(2026, 8, 25),
            http_get=get,
            cache={},
        )

        self.assertEqual(result["symbol"], "TAIEX")
        self.assertEqual(result["as_of"], "2026-08-25")
        self.assertEqual(result["candles"][-1]["time"], "2026-08-25")
        self.assertGreaterEqual(len(result["candles"]), 60)
        self.assertTrue(result["ma20"])
        self.assertEqual(len(calls), 12)
        self.assertEqual(result["source"], "臺灣證券交易所")

    def test_snapshot_fails_closed_when_official_latest_date_does_not_match(self):
        def get(_url, *, params, headers, timeout):
            month = params["date"]
            year = int(month[:4])
            month_number = int(month[4:6])
            return _Response(
                _payload(
                    month,
                    [[f"{year - 1911:03d}/{month_number:02d}/01", "100", "110", "90", "105"]],
                )
            )

        self.assertIsNone(
            fetch_twse_index_snapshot(
                datetime.date(2026, 8, 25),
                http_get=get,
                cache={},
            )
        )


if __name__ == "__main__":
    unittest.main()
