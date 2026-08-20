"""Tests for US market calendar and session boundary resolution."""

import datetime
import unittest
import zoneinfo

from stock_papi.batch.calendar import CalendarError, TradingCalendar, TradingCalendarSet
from stock_papi.integrations.market_data.us_calendar import (
    generate_us_calendar_document,
    get_us_calendar_documents,
    get_us_exchange_holidays,
)

NEW_YORK = zoneinfo.ZoneInfo("America/New_York")


class USCalendarTests(unittest.TestCase):
    def test_2026_us_calendar_document_valid(self):
        doc = generate_us_calendar_document(2026)
        self.assertEqual(doc["schema_version"], 1)
        self.assertEqual(doc["market"], "US")
        self.assertEqual(doc["year"], 2026)
        self.assertIn("2026-01-01", doc["closed_dates"])  # New Year's Day
        self.assertIn("2026-01-19", doc["closed_dates"])  # MLK Day
        self.assertIn("2026-02-16", doc["closed_dates"])  # Presidents' Day
        self.assertIn("2026-04-03", doc["closed_dates"])  # Good Friday
        self.assertIn("2026-05-25", doc["closed_dates"])  # Memorial Day
        self.assertIn("2026-06-19", doc["closed_dates"])  # Juneteenth
        self.assertIn("2026-07-03", doc["closed_dates"])  # Independence Day observed
        self.assertIn("2026-09-07", doc["closed_dates"])  # Labor Day
        self.assertIn("2026-11-26", doc["closed_dates"])  # Thanksgiving Day
        self.assertIn("2026-12-25", doc["closed_dates"])  # Christmas Day
        self.assertIn("2026-11-27", doc["early_closed_dates"])  # Black Friday
        self.assertIn("2026-12-24", doc["early_closed_dates"])  # Christmas Eve

        cal = TradingCalendar.from_document(doc)
        self.assertEqual(cal.market, "US")
        self.assertFalse(cal.is_session(datetime.date(2026, 1, 1)))
        self.assertFalse(cal.is_session(datetime.date(2026, 1, 3)))  # Saturday
        self.assertFalse(cal.is_session(datetime.date(2026, 1, 4)))  # Sunday
        self.assertTrue(cal.is_session(datetime.date(2026, 1, 2)))  # Friday session
        self.assertTrue(cal.is_early_close(datetime.date(2026, 11, 27)))
        self.assertFalse(cal.is_early_close(datetime.date(2026, 1, 2)))

    def test_multi_year_calendar_set(self):
        docs = get_us_calendar_documents(2025, 2027)
        cal_set = TradingCalendarSet.from_documents(docs)
        # Next session after 2026-08-14 (Fri) is 2026-08-17 (Mon)
        self.assertEqual(
            cal_set.next_session(datetime.date(2026, 8, 14)),
            datetime.date(2026, 8, 17),
        )
        # Next session after 2026-08-19 (Wed) is 2026-08-20 (Thu)
        self.assertEqual(
            cal_set.next_session(datetime.date(2026, 8, 19)),
            datetime.date(2026, 8, 20),
        )
        # Next session before Thanksgiving (2026-11-26 Thu) -> 2026-11-27 (Fri, early close)
        self.assertEqual(
            cal_set.next_session(datetime.date(2026, 11, 25)),
            datetime.date(2026, 11, 27),
        )

    def test_invalid_market_in_calendar(self):
        doc = generate_us_calendar_document(2026)
        doc["market"] = "INVALID"
        with self.assertRaises(CalendarError):
            TradingCalendar.from_document(doc)

    def test_overlap_closed_and_early(self):
        doc = generate_us_calendar_document(2026)
        doc["early_closed_dates"].append(doc["closed_dates"][0])
        with self.assertRaises(CalendarError):
            TradingCalendar.from_document(doc)
