"""Tests for US trading status and halt evidence contracts."""

import datetime
import hashlib
import json
import unittest

from stock_papi.integrations.market_data.us_trading_status import (
    create_us_status_evidence,
    evidence_sha256,
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
            raw_fields={"reason": "T1", "halt_date": "2026-08-19"},
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


if __name__ == "__main__":
    unittest.main()
