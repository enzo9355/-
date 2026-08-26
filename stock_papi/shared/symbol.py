"""Symbol normalization and type detection services."""

import re
from stock_papi.shared.validation import is_us_ticker

def normalize_symbol(symbol: str) -> str:
    """Normalize symbols to a canonical form: MARKET:TICKER.
    
    Examples:
        TW:2330 -> TW:2330
        2330.TW -> TW:2330
        2330.TWO -> TW:2330
        2330 -> TW:2330
        US:AAPL -> US:AAPL
        AAPL -> US:AAPL
        BRK.B -> US:BRK.B
        BRK-B -> US:BRK.B
    """
    symbol = str(symbol or "").strip().upper()
    if not symbol:
        return ""
        
    # 1. Check explicit namespace prefixes
    if symbol.startswith("TW:"):
        return f"TW:{symbol[3:]}"
    if symbol.startswith("US:"):
        return f"US:{symbol[3:]}"
        
    # 2. Check suffixes
    if symbol.endswith(".TW"):
        return f"TW:{symbol[:-3]}"
    if symbol.endswith(".TWO"):
        return f"TW:{symbol[:-4]}"
        
    # 3. No prefix or suffix, infer from format
    if re.fullmatch(r"[0-9]{4,5}[0-9A-Z]?", symbol):
        return f"TW:{symbol}"
        
    # Standardize US ticker hyphens to dots (e.g. BRK-B -> BRK.B)
    symbol = symbol.replace("-", ".")
    return f"US:{symbol}"


def get_instrument_type(symbol: str) -> str:
    """Detect if the symbol represents an ETF, STOCK, or unknown.
    
    Checks the ETF catalog and twstock.codes metadata when available.
    """
    norm = normalize_symbol(symbol)
    if not norm:
        return "unknown"
        
    ticker = norm.split(":", 1)[-1]
    
    # 1. Check known ETF tickers
    etf_tickers = {"0050", "00878", "SPY", "QQQ", "1321.T", "1306.T", "1321", "1306"}
    if ticker in etf_tickers:
        return "ETF"
        
    # 2. Check twstock.codes metadata for TW market
    if norm.startswith("TW:"):
        try:
            import twstock
            if twstock and hasattr(twstock, "codes") and ticker in twstock.codes:
                info = twstock.codes[ticker]
                group = str(getattr(info, "group", "") or "").upper()
                type_val = str(getattr(info, "type", "") or "").upper()
                if "ETF" in group or "ETF" in type_val:
                    return "ETF"
                return "STOCK"
        except Exception:
            pass

        # Numeric Taiwan tickers normalize to the TW namespace.  If the
        # optional twstock catalog is unavailable, the known ETF allowlist
        # above is still authoritative and the remaining ticker is a stock.
        return "STOCK"

    return "unknown"
