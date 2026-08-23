"""Tests for US stock market universe from SEC exchange listings."""

import datetime
import tempfile
import unittest
from unittest.mock import patch

from stock_papi.integrations.market_data.us_universe import (
    get_us_symbols,
    parse_sec_us_universe,
    validate_us_ticker,
)


class USUniverseTests(unittest.TestCase):
    def test_validate_us_ticker(self):
        self.assertEqual(validate_us_ticker("AAPL"), "AAPL")
        self.assertEqual(validate_us_ticker("BRK-B"), "BRK-B")
        self.assertEqual(validate_us_ticker("nvda"), "NVDA")
        self.assertEqual(validate_us_ticker("SPY"), "SPY")

        with self.assertRaises(ValueError):
            validate_us_ticker("")
        with self.assertRaises(ValueError):
            validate_us_ticker("../traversal")
        with self.assertRaises(ValueError):
            validate_us_ticker("A" * 15)
        with self.assertRaises(ValueError):
            validate_us_ticker("INVALID/SLASH")

    def test_parse_sec_us_universe(self):
        doc = {
            "fields": ["cik", "name", "ticker", "exchange"],
            "data": [
                [320193, "Apple Inc.", "AAPL", "Nasdaq"],
                [1045810, "NVIDIA CORP", "NVDA", "Nasdaq"],
                [1067983, "BERKSHIRE HATHAWAY INC", "BRK.B", "NYSE"],
                [1234567, "Some Penny Stock", "OTCXYZ", "OTC"],
                [9999999, "Bitcoin Trust Fund", "BTCF", "Nasdaq"],
            ],
        }
        symbols = parse_sec_us_universe(doc)
        self.assertEqual(symbols, ["AAPL", "BRK-B", "NVDA"])
        self.assertNotIn("OTCXYZ", symbols)  # OTC excluded
        self.assertNotIn("BTCF", symbols)  # Crypto excluded

    def test_get_us_symbols_cache_and_fallback(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            sample_doc = {
                "fields": ["cik", "name", "ticker", "exchange"],
                "data": [
                    [1, "Apple Inc.", "AAPL", "Nasdaq"],
                    [2, "Microsoft Corp", "MSFT", "Nasdaq"],
                ],
            }
            # First fetch (creates cache)
            symbols = get_us_symbols(
                tmpdir,
                fetch_json=lambda: sample_doc,
                now=datetime.datetime(2026, 8, 20, tzinfo=datetime.timezone.utc),
            )
            self.assertEqual(symbols, ["AAPL", "MSFT"])

            # Second fetch with failing source (uses cache)
            def failing_fetch():
                raise RuntimeError("network down")

            symbols2 = get_us_symbols(
                tmpdir,
                fetch_json=failing_fetch,
                now=datetime.datetime(2026, 8, 20, tzinfo=datetime.timezone.utc),
            )
            self.assertEqual(symbols2, ["AAPL", "MSFT"])

    def test_derivative_security_type_exclusions(self):
        from stock_papi.integrations.market_data.us_universe import parse_sec_us_universe_with_metadata
        doc = {
            "fields": ["cik", "name", "ticker", "exchange"],
            "data": [
                [1, "Apple Inc.", "AAPL", "Nasdaq"],
                [2, "SPDR S&P 500 ETF Trust", "SPY", "NYSE"],
                [3, "Some Acquisition Warrant", "AACIW", "Nasdaq"],
                [4, "Some SPAC Unit", "AAC-UN", "Nasdaq"],
                [5, "Some Company Preferred Stock", "ACP-PA", "NYSE"],
                [6, "Some Subscription Right", "AESPR", "Nasdaq"],
            ],
        }
        security_metadata = {
            "AACIW": {"security_type": "WARRANT", "source_id": "nasdaqtrader:nasdaqlisted", "as_of": "2026-08-20", "evidence_sha256": "a" * 64},
            "AAC-UN": {"security_type": "UNIT", "source_id": "nasdaqtrader:nasdaqlisted", "as_of": "2026-08-20", "evidence_sha256": "b" * 64},
            "ACP-PA": {"security_type": "PREFERRED", "source_id": "nasdaqtrader:otherlisted", "as_of": "2026-08-20", "evidence_sha256": "c" * 64},
            "AESPR": {"security_type": "RIGHT", "source_id": "nasdaqtrader:nasdaqlisted", "as_of": "2026-08-20", "evidence_sha256": "d" * 64},
        }
        # Equity observation scope excludes derivative instruments
        equity_bd = parse_sec_us_universe_with_metadata(
            doc, scope="EQUITY_OBSERVATION", security_metadata=security_metadata
        )
        self.assertEqual(equity_bd.symbols, ["AAPL", "SPY"])
        self.assertEqual(equity_bd.excluded_derivative_count, 4)
        self.assertEqual(equity_bd.derivative_breakdown["WARRANT"], 1)
        self.assertEqual(equity_bd.derivative_breakdown["UNIT"], 1)
        self.assertEqual(equity_bd.derivative_breakdown["PREFERRED"], 1)
        self.assertEqual(equity_bd.derivative_breakdown["RIGHT"], 1)

        # ALL_LISTED scope includes derivatives
        all_bd = parse_sec_us_universe_with_metadata(doc, scope="ALL_LISTED")
        self.assertEqual(len(all_bd.symbols), 6)
        self.assertEqual(all_bd.excluded_derivative_count, 0)

    def test_suffix_only_signals_do_not_remove_symbols_from_denominator(self):
        from stock_papi.integrations.market_data.us_universe import parse_sec_us_universe_with_metadata
        doc = {
            "fields": ["cik", "name", "ticker", "exchange"],
            "data": [
                [1, "Looks Like A Warrant", "AACIW", "Nasdaq"],
                [2, "Looks Like A Right", "AESPR", "Nasdaq"],
                [3, "Looks Like A Unit", "AACIU", "Nasdaq"],
                [4, "Looks Like Preferred", "ACP-PA", "NYSE"],
            ],
        }
        breakdown = parse_sec_us_universe_with_metadata(doc, scope="EQUITY_OBSERVATION")
        self.assertEqual(breakdown.symbols, ["AACIU", "AACIW", "ACP-PA", "AESPR"])
        self.assertEqual(breakdown.excluded_derivative_count, 0)
        self.assertEqual(breakdown.security_metadata_status, "unavailable")
        self.assertEqual(
            breakdown.security_eligibility_by_symbol["AACIW"]["heuristic_signal"],
            "WARRANT",
        )

    def test_lifecycle_evidence_excludes_only_effective_events(self):
        from stock_papi.integrations.market_data.us_universe import parse_sec_us_universe_with_metadata
        doc = {
            "fields": ["cik", "name", "ticker", "exchange"],
            "data": [
                [1, "Old Corp - Common Stock", "OLD", "Nasdaq"],
                [2, "Live Corp - Common Stock", "LIVE", "Nasdaq"],
            ],
        }
        events = [
            {
                "symbol": "OLD",
                "event": "delisted",
                "effective_date": "2026-08-18",
                "source": "exchange-daily-list",
                "source_identity": "exchange-daily-list:2026-08-18",
                "evidence_sha256": "e" * 64,
            },
            {
                "symbol": "LIVE",
                "event": "delisted",
                "effective_date": "2026-08-22",
                "source": "exchange-daily-list",
                "source_identity": "exchange-daily-list:2026-08-22",
                "evidence_sha256": "f" * 64,
            },
        ]
        breakdown = parse_sec_us_universe_with_metadata(
            doc,
            target_market_date=datetime.date(2026, 8, 20),
            lifecycle_events=events,
        )
        self.assertEqual(breakdown.symbols, ["LIVE"])
        self.assertEqual(breakdown.terminated_delisted_count, 1)
        self.assertEqual(breakdown.lifecycle_evidence_status, "available")
        self.assertEqual(breakdown.lifecycle_events_by_symbol["OLD"]["event"], "delisted")
        self.assertEqual(breakdown.exclusions_by_symbol["OLD"]["source_identity"], "exchange-daily-list:2026-08-18")

    def test_lifecycle_count_is_not_fabricated_when_source_is_unavailable(self):
        from stock_papi.integrations.market_data.us_universe import parse_sec_us_universe_with_metadata
        doc = {
            "fields": ["cik", "name", "ticker", "exchange"],
            "data": [[1, "Live Corp - Common Stock", "LIVE", "Nasdaq"]],
        }
        breakdown = parse_sec_us_universe_with_metadata(doc)
        self.assertIsNone(breakdown.terminated_delisted_count)
        self.assertEqual(breakdown.lifecycle_evidence_status, "lifecycle_evidence_unavailable")

    def test_first_party_directory_metadata_classifies_supported_security_types(self):
        from stock_papi.integrations.market_data.us_universe import (
            parse_nasdaq_security_directory,
            parse_sec_us_universe_with_metadata,
        )
        directory = parse_nasdaq_security_directory(
            "\n".join(
                [
                    "Symbol|Security Name|Market Category|Test Issue|Financial Status|Round Lot Size|ETF|NextShares",
                    "AAPL|Apple Inc. - Common Stock|Q|N|N|100|N|N",
                    "AADR|AdvisorShares Dorsey Wright ADR ETF|G|N|N|100|Y|N",
                    "AACIW|Example Acquisition - Warrant|G|N|N|100|N|N",
                    "ACP-PA|Example Corp - Preferred Stock|Q|N|N|100|N|N",
                    "File Creation Time: 0820202616:00|||||||",
                ]
            ),
            source_id="nasdaqtrader:nasdaqlisted",
            source_url="https://example.test/nasdaqlisted.txt",
        )
        self.assertEqual(directory["as_of"], "2026-08-20")
        self.assertEqual(directory["records"]["AADR"]["etf"], "Y")
        self.assertEqual(directory["records"]["AAPL"]["source_id"], "nasdaqtrader:nasdaqlisted")

        sec_doc = {
            "fields": ["cik", "name", "ticker", "exchange"],
            "data": [
                [1, "Apple Inc.", "AAPL", "Nasdaq"],
                [2, "AdvisorShares", "AADR", "Nasdaq"],
                [3, "Example Acquisition", "AACIW", "Nasdaq"],
                [4, "Example Corp", "ACP-PA", "Nasdaq"],
            ],
        }
        breakdown = parse_sec_us_universe_with_metadata(
            sec_doc,
            security_metadata=directory["records"],
        )
        self.assertEqual(breakdown.security_eligibility_by_symbol["AAPL"]["security_type"], "COMMON_EQUITY")
        self.assertEqual(breakdown.security_eligibility_by_symbol["AADR"]["security_type"], "ETF")
        self.assertEqual(breakdown.security_eligibility_by_symbol["AACIW"]["security_type"], "WARRANT")
        self.assertEqual(breakdown.security_eligibility_by_symbol["ACP-PA"]["security_type"], "PREFERRED")
        self.assertNotIn("AACIW", breakdown.symbols)
        self.assertEqual(breakdown.exclusions_by_symbol["AACIW"]["source_id"] if "source_id" in breakdown.exclusions_by_symbol["AACIW"] else breakdown.exclusions_by_symbol["AACIW"]["source"], "nasdaqtrader:nasdaqlisted")

    def test_nasdaq_etn_is_authoritatively_excluded_from_equity_observation(self):
        from stock_papi.integrations.market_data.us_universe import (
            parse_nasdaq_security_directory,
            parse_sec_us_universe_with_metadata,
        )

        directory = parse_nasdaq_security_directory(
            "\n".join(
                [
                    "ACT Symbol|Security Name|Exchange|CQS Symbol|ETF|Round Lot Size|Test Issue|NASDAQ Symbol",
                    "BRZL|MicroSectors 3x Long Brazil ETNs|Z|BRZL|Y|100|N|BRZL",
                    "File Creation Time: 0821202616:00|||||||",
                ]
            ),
            source_id="nasdaqtrader:otherlisted",
            source_url="https://example.test/otherlisted.txt",
        )
        self.assertEqual(directory["records"]["BRZL"]["security_type"], "ETN")

        breakdown = parse_sec_us_universe_with_metadata(
            {
                "fields": ["cik", "name", "ticker", "exchange"],
                "data": [
                    [1, "MicroSectors 3x Long Brazil ETNs", "BRZL", "Nasdaq"],
                    [2, "Apple Inc. Common Stock", "AAPL", "Nasdaq"],
                ],
            },
            security_metadata=directory["records"],
        )
        self.assertEqual(breakdown.symbols, ["AAPL"])
        self.assertEqual(breakdown.excluded_derivative_count, 1)
        self.assertEqual(breakdown.derivative_breakdown["ETN"], 1)
        self.assertEqual(
            breakdown.exclusions_by_symbol["BRZL"]["reason"],
            "excluded_authoritative_security_type_etn",
        )

    def test_nasdaq_etn_wins_generic_cross_exchange_etf_conflict(self):
        from stock_papi.integrations.market_data.us_universe import fetch_official_us_security_metadata

        nasdaq_record = {
            "BRZL": {
                "security_name": "MicroSectors 3x Long Brazil ETNs",
                "security_type": "ETN",
                "source_id": "nasdaqtrader:otherlisted",
                "as_of": "2026-08-21",
            }
        }
        nyse_record = {
            "BRZL": {
                "security_name": "MicroSectors 3x Long Brazil ETF",
                "security_type": "ETF",
                "source_id": "nyse:security_mapping",
                "as_of": "2026-08-24",
            }
        }
        with patch(
            "stock_papi.integrations.market_data.us_universe.fetch_nasdaq_security_directory",
            return_value={"records": nasdaq_record, "sources": [], "status": "healthy"},
        ), patch(
            "stock_papi.integrations.market_data.us_universe.fetch_nyse_security_mapping",
            return_value={"records": nyse_record, "sources": [], "status": "healthy"},
        ):
            metadata = fetch_official_us_security_metadata(
                target_market_date=datetime.date(2026, 8, 21)
            )
        self.assertEqual(metadata["records"]["BRZL"]["security_type"], "ETN")
        self.assertNotIn("BRZL", metadata["conflicted_symbols"])

    def test_nasdaq_corporate_action_parser_excludes_only_claimed_symbols(self):
        from stock_papi.integrations.market_data.us_universe import parse_nasdaq_corporate_action_alert

        text = """
        Company Name/Issue: Inflection Point Acquisition Corp. III Class A Ordinary Shares
        CUSIP#: G47875102 Symbol: IPCX Last Trading Date: August 14, 2026
        Marketplace Effective Date for Suspension: August 17, 2026
        Company Name/Issue: Inflection Point Acquisition Corp. III Rights
        CUSIP#: G47875110 Symbol: IPCXR Last Trading Date: August 14, 2026
        Marketplace Effective Date for Suspension: August 17, 2026
        Company Name/Issue: Air Water Ventures Limited Ordinary Shares
        CUSIP#: 00000000 Symbol: WATR Marketplace Effective Date: August 17, 2026
        """
        document = parse_nasdaq_corporate_action_alert(text)
        self.assertEqual(
            [(event["symbol"], event["security_type"]) for event in document["events"]],
            [("IPCX", "COMMON_EQUITY"), ("IPCXR", "RIGHT")],
        )

    def test_effective_lifecycle_events_use_event_security_identity_and_preserve_replacement(self):
        from stock_papi.integrations.market_data.us_universe import parse_sec_us_universe_with_metadata

        doc = {
            "fields": ["cik", "name", "ticker", "exchange"],
            "data": [
                [1, "iShares IPCX Class A Ordinary Shares", "IPCX", "Nasdaq"],
                [2, "iShares IPCX Rights", "IPCXR", "Nasdaq"],
                [3, "iShares IPCX Units", "IPCXU", "Nasdaq"],
                [4, "WATER Technologies Common Stock", "WATR", "Nasdaq"],
            ],
        }
        source_identity = "nasdaqtrader:corporate_action:ECA2026-576:" + "a" * 64
        events = [
            {
                "symbol": "IPCX",
                "event": "suspended",
                "effective_date": "2026-08-17",
                "last_trading_date": "2026-08-14",
                "security_type": "COMMON_EQUITY",
                "source": "https://www.nasdaqtrader.com/TraderNews.aspx?id=ECA2026-576",
                "source_identity": source_identity,
                "payload_sha256": "a" * 64,
                "evidence_sha256": "b" * 64,
            },
            {
                "symbol": "IPCXR",
                "event": "suspended",
                "effective_date": "2026-08-17",
                "last_trading_date": "2026-08-14",
                "security_type": "RIGHT",
                "source": "https://www.nasdaqtrader.com/TraderNews.aspx?id=ECA2026-576",
                "source_identity": source_identity,
                "payload_sha256": "a" * 64,
                "evidence_sha256": "c" * 64,
            },
            {
                "symbol": "IPCXU",
                "event": "suspended",
                "effective_date": "2026-08-17",
                "last_trading_date": "2026-08-14",
                "security_type": "UNIT",
                "source": "https://www.nasdaqtrader.com/TraderNews.aspx?id=ECA2026-576",
                "source_identity": source_identity,
                "payload_sha256": "a" * 64,
                "evidence_sha256": "d" * 64,
            },
        ]

        before_effective = parse_sec_us_universe_with_metadata(
            doc,
            target_market_date=datetime.date(2026, 8, 14),
            lifecycle_events=events,
        )
        self.assertIn("IPCX", before_effective.symbols)
        self.assertIn("IPCXR", before_effective.symbols)
        self.assertIn("IPCXU", before_effective.symbols)
        self.assertIn("WATR", before_effective.symbols)

        after_effective = parse_sec_us_universe_with_metadata(
            doc,
            target_market_date=datetime.date(2026, 8, 21),
            lifecycle_events=events,
        )
        self.assertNotIn("IPCX", after_effective.symbols)
        self.assertNotIn("IPCXR", after_effective.symbols)
        self.assertNotIn("IPCXU", after_effective.symbols)
        self.assertIn("WATR", after_effective.symbols)
        self.assertEqual(after_effective.exclusions_by_symbol["IPCX"]["event"], "suspended")
        self.assertEqual(after_effective.exclusions_by_symbol["IPCXR"]["classification"], "RIGHT")
        self.assertEqual(after_effective.security_eligibility_by_symbol["IPCX"]["security_type"], "COMMON_EQUITY")

    def test_first_party_directory_normalizes_official_security_aliases(self):
        from stock_papi.integrations.market_data.us_universe import parse_nasdaq_security_directory

        directory = parse_nasdaq_security_directory(
            "\n".join(
                [
                    "ACT Symbol|Security Name|Exchange|CQS Symbol|ETF|Round Lot Size|Test Issue|NASDAQ Symbol",
                    "ABR$D|Example Corp Preferred Stock|N|ABRpD|N|100|N|ABR-D",
                    "AAC.U|Example Acquisition Units|N|AAC.U|N|100|N|AAC=",
                    "ACHR.W|Example Acquisition Warrants|N|ACHR.WS|N|100|N|ACHR+",
                    "NE.A|Example Tranche 2 Warrants|N|NE.A|N|100|N|NE^",
                    "File Creation Time: 0820202616:00|||||||",
                ]
            ),
            source_id="nasdaqtrader:otherlisted",
            source_url="https://example.test/otherlisted.txt",
        )
        self.assertEqual(directory["records"]["ABR-PD"]["security_type"], "PREFERRED")
        self.assertEqual(directory["records"]["AAC-UN"]["security_type"], "UNIT")
        self.assertEqual(directory["records"]["ACHR-WT"]["security_type"], "WARRANT")
        self.assertEqual(directory["records"]["NE-WTA"]["security_type"], "WARRANT")

    def test_nyse_security_mapping_uses_typed_codes_and_aliases(self):
        from stock_papi.integrations.market_data.us_universe import parse_nyse_security_mapping

        directory = parse_nyse_security_mapping(
            """<NYSESymbolMap>
              <SymbolMap><Symbol>ABR PRF</Symbol><CQS_Symbol>ABRpF</CQS_Symbol><ListedMarket>N</ListedMarket><Security_Type>P</Security_Type></SymbolMap>
              <SymbolMap><Symbol>AAC U</Symbol><CQS_Symbol>AAC.U</CQS_Symbol><ListedMarket>N</ListedMarket><Security_Type>I</Security_Type></SymbolMap>
              <SymbolMap><Symbol>ACHR WS</Symbol><CQS_Symbol>ACHR.WS</CQS_Symbol><ListedMarket>N</ListedMarket><Security_Type>W</Security_Type></SymbolMap>
              <SymbolMap><Symbol>AAPL</Symbol><CQS_Symbol>AAPL</CQS_Symbol><ListedMarket>Q</ListedMarket><Security_Type>C</Security_Type></SymbolMap>
            </NYSESymbolMap>""",
            source_id="nyse:security_mapping",
            source_url="https://example.test/NYSESymbolMapping_20260820.xml",
        )
        self.assertEqual(directory["records"]["ABR-PF"]["security_type"], "PREFERRED")
        self.assertEqual(directory["records"]["AAC-UN"]["security_type"], "UNIT")
        self.assertEqual(directory["records"]["ACHR-WT"]["security_type"], "WARRANT")
        self.assertEqual(directory["records"]["AAPL"]["security_type"], "COMMON_EQUITY")
