"""Tests for US trading status and halt evidence contracts."""

import datetime
import hashlib
import json
import unittest
from unittest.mock import patch

from stock_papi.integrations.market_data.us_trading_status import (
    create_us_status_evidence,
    evidence_sha256,
    fetch_nasdaq_trade_halts,
    is_halt_effective_for_target_session,
    USStatusOperationalError,
    USStatusSchemaError,
    validate_us_status_evidence,
)


class TestUSTradingStatus(unittest.TestCase):
    def setUp(self):
        self.target_date = datetime.date(2026, 8, 19)

    def test_create_and_validate_halt_evidence(self):
        payload_bytes = b'{"halt_time": "10:00:00", "reason": "T1"}'
        payload_hash = hashlib.sha256(payload_bytes).hexdigest()

        evidence = create_us_status_evidence(
            status="officially_suspended",
            symbol="AAPL",
            target_market_date=self.target_date,
            exchange="NASDAQ",
            source_id="nasdaq_halts",
            payload_sha256=payload_hash,
            raw_fields={
                "reason": "T1",
                "halt_date": "2026-08-19",
                "effective_on_target_session": True,
            },
        )

        self.assertEqual(evidence["market"], "US")
        self.assertEqual(evidence["symbol"], "AAPL")
        self.assertEqual(evidence["status"], "officially_suspended")
        self.assertEqual(evidence["evidence_sha256"], evidence_sha256(evidence))

        # Validate with matching symbol & target_date
        validated = validate_us_status_evidence(
            evidence, symbol="AAPL", target_date=self.target_date
        )
        self.assertEqual(validated, evidence)

    def test_tampered_evidence_fails_validation(self):
        payload_hash = hashlib.sha256(b"raw").hexdigest()
        evidence = create_us_status_evidence(
            status="officially_suspended",
            symbol="MSFT",
            target_market_date=self.target_date,
            exchange="NASDAQ",
            source_id="nasdaq_halts",
            payload_sha256=payload_hash,
            raw_fields={"effective_on_target_session": True},
        )

        # Tamper symbol
        tampered = dict(evidence)
        tampered["symbol"] = "NVDA"
        with self.assertRaises(ValueError):
            validate_us_status_evidence(tampered, symbol="NVDA", target_date=self.target_date)

        # Tamper hash
        tampered2 = dict(evidence)
        tampered2["evidence_sha256"] = "0" * 64
        with self.assertRaises(ValueError):
            validate_us_status_evidence(tampered2, symbol="MSFT", target_date=self.target_date)

    def test_unsupported_exchange_fails(self):
        payload_hash = hashlib.sha256(b"raw").hexdigest()
        with self.assertRaises(ValueError):
            create_us_status_evidence(
                status="officially_suspended",
                symbol="MSFT",
                target_market_date=self.target_date,
                exchange="UNKNOWN_EXCHANGE",
                source_id="nasdaq_halts",
                payload_sha256=payload_hash,
            )

    def test_network_failure_is_not_empty_status_evidence(self):
        with patch("urllib.request.urlopen", side_effect=TimeoutError("timed out")):
            with self.assertRaises(USStatusOperationalError):
                fetch_nasdaq_trade_halts(self.target_date)

    def test_malformed_xml_is_schema_failure(self):
        with self.assertRaises(USStatusSchemaError):
            fetch_nasdaq_trade_halts(self.target_date, mock_xml="<rss><channel>")

    def test_valid_feed_with_no_matching_halts_returns_empty_map(self):
        xml = "<rss><channel><title>Trade Halts</title></channel></rss>"
        self.assertEqual(fetch_nasdaq_trade_halts(self.target_date, mock_xml=xml), {})

    def test_target_session_halt_feed_creates_hash_bound_effective_evidence(self):
        xml = """<rss><channel><item><description><![CDATA[
        <table><tr><td>AAPL</td><td>Apple Inc.</td><td>NASDAQ</td><td>T1</td>
        <td>08/19/2026</td><td>09:30:00</td></tr></table>
        ]]></description></item></channel></rss>"""
        result = fetch_nasdaq_trade_halts(self.target_date, mock_xml=xml)
        self.assertIn("AAPL", result)
        self.assertTrue(result["AAPL"]["raw_fields"]["effective_on_target_session"])
        self.assertEqual(result["AAPL"]["target_market_date"], "2026-08-19")

    def test_official_namespaced_nasdaq_feed_schema_is_parsed(self):
        xml = """<?xml version="1.0"?>
        <rss version="2.0" xmlns:ndaq="http://www.nasdaqtrader.com/">
          <channel><title>NASDAQTrader.com</title>
            <item>
              <title>AAPL</title>
              <ndaq:HaltDate>08/19/2026</ndaq:HaltDate>
              <ndaq:HaltTime>09:30:00.000</ndaq:HaltTime>
              <ndaq:IssueSymbol>AAPL</ndaq:IssueSymbol>
              <ndaq:IssueName>Apple Inc. CS</ndaq:IssueName>
              <ndaq:Market>NASDAQ</ndaq:Market>
              <ndaq:ReasonCode>T1</ndaq:ReasonCode>
              <ndaq:ResumptionDate />
              <ndaq:ResumptionQuoteTime />
              <ndaq:ResumptionTradeTime />
              <description><![CDATA[<table><tr>
                <th>Halt Date</th><th>Halt Time</th><th>Issue Symbol</th>
                <th>Issue Name</th><th>Market</th><th>Reason Code</th>
                <th>Pause Threshold Price</th><th>Resumption Date</th>
                <th>Resumption Quote Time</th><th>Resumption Trade Time</th>
              </tr><tr>
                <td>08/19/2026</td><td>09:30:00.000</td><td>AAPL</td>
                <td>Apple Inc. CS</td><td>NASDAQ</td><td>T1</td>
                <td></td><td></td><td></td><td></td>
              </tr></table>]]></description>
            </item>
          </channel>
        </rss>"""
        result = fetch_nasdaq_trade_halts(self.target_date, mock_xml=xml)
        self.assertIn("AAPL", result)
        self.assertEqual(result["AAPL"]["raw_fields"]["reason_code"], "T1")
        self.assertEqual(result["AAPL"]["raw_fields"]["halt_time"], "09:30:00")

    def test_fractional_second_status_time_is_supported(self):
        self.assertTrue(
            is_halt_effective_for_target_session(
                halt_date=self.target_date,
                target_market_date=self.target_date,
                halt_time="09:30:00.000",
            )
        )

    def test_resumed_historical_halt_is_not_emitted_as_target_status(self):
        xml = """<rss><channel><item><description><![CDATA[
        <table><tr><td>AAPL</td><td>Apple Inc.</td><td>NASDAQ</td><td>T1</td>
        <td>08/17/2026</td><td>09:30:00</td><td>08/18/2026</td><td>16:00:00</td>
        </tr></table>
        ]]></description></item></channel></rss>"""
        self.assertEqual(fetch_nasdaq_trade_halts(self.target_date, mock_xml=xml), {})

    def test_halt_before_target_resumed_before_target_is_not_effective(self):
        self.assertFalse(
            is_halt_effective_for_target_session(
                halt_date=datetime.date(2026, 8, 17),
                target_market_date=self.target_date,
                resumption_date=datetime.date(2026, 8, 18),
            )
        )

    def test_halt_on_target_without_resumption_is_effective(self):
        self.assertTrue(
            is_halt_effective_for_target_session(
                halt_date=self.target_date,
                target_market_date=self.target_date,
            )
        )

    def test_halt_resumed_before_regular_close_is_not_full_session_halt(self):
        self.assertFalse(
            is_halt_effective_for_target_session(
                halt_date=self.target_date,
                target_market_date=self.target_date,
                resumption_date=self.target_date,
                resumption_time="12:00",
            )
        )

    def test_historical_halt_without_target_session_evidence_is_not_automatically_effective(self):
        self.assertFalse(
            is_halt_effective_for_target_session(
                halt_date=datetime.date(2026, 8, 1),
                target_market_date=self.target_date,
                resumption_date=datetime.date(2026, 8, 2),
            )
        )


if __name__ == "__main__":
    unittest.main()
