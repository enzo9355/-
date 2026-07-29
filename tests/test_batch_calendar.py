import datetime
import unittest


from stock_papi.batch.calendar import CalendarError, TradingCalendarSet


def calendar_document(year, *, closed=(), special_open=()):
    return {
        "schema_version": 1,
        "market": "TW",
        "year": year,
        "source_url": "https://openapi.twse.com.tw/v1/holidaySchedule/holidaySchedule",
        "fetched_at": f"{year - 1}-12-01T00:00:00Z",
        "source_sha256": "a" * 64,
        "valid_from": f"{year}-01-01",
        "valid_to": f"{year}-12-31",
        "closed_dates": list(closed),
        "special_open_dates": list(special_open),
    }


class TradingCalendarTests(unittest.TestCase):
    def test_official_closures_and_special_open_days_override_weekdays(self):
        calendars = TradingCalendarSet.from_documents([
            calendar_document(
                2026,
                closed=("2026-02-16", "2026-02-17"),
                special_open=("2026-02-14",),
            )
        ])

        self.assertTrue(calendars.is_session(datetime.date(2026, 2, 13)))
        self.assertFalse(calendars.is_session(datetime.date(2026, 2, 16)))
        self.assertTrue(calendars.is_session(datetime.date(2026, 2, 14)))
        self.assertFalse(calendars.is_session(datetime.date(2026, 2, 15)))

    def test_missing_year_or_invalid_source_fails_closed(self):
        calendars = TradingCalendarSet.from_documents([calendar_document(2026)])

        with self.assertRaises(CalendarError):
            calendars.is_session(datetime.date(2027, 1, 4))

        invalid = calendar_document(2026)
        invalid["source_url"] = "https://example.com/calendar.json"
        with self.assertRaises(CalendarError):
            TradingCalendarSet.from_documents([invalid])

    def test_next_session_and_five_session_horizon_cross_holiday(self):
        calendars = TradingCalendarSet.from_documents([
            calendar_document(2026, closed=("2026-07-20",))
        ])
        source = datetime.date(2026, 7, 17)

        applicable = calendars.next_session(source)

        self.assertEqual(applicable, datetime.date(2026, 7, 21))
        self.assertEqual(
            calendars.session_offset(applicable, 4),
            datetime.date(2026, 7, 27),
        )

    def test_calendar_rejects_overlapping_or_out_of_year_dates(self):
        overlapping = calendar_document(
            2026,
            closed=("2026-02-16",),
            special_open=("2026-02-16",),
        )
        with self.assertRaises(CalendarError):
            TradingCalendarSet.from_documents([overlapping])

        outside = calendar_document(2026, closed=("2027-01-01",))
        with self.assertRaises(CalendarError):
            TradingCalendarSet.from_documents([outside])

    def test_calendar_accepts_rfc3339_fraction_longer_than_microseconds(self):
        document = calendar_document(2026)
        document["fetched_at"] = "2026-07-15T13:49:37.9260379+00:00"

        calendars = TradingCalendarSet.from_documents([document])

        self.assertTrue(calendars.is_session(datetime.date(2026, 7, 15)))


if __name__ == "__main__":
    unittest.main()

