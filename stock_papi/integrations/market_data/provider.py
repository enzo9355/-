"""FinMind, Yahoo Finance and ticker lookup adapters."""

import datetime


class FinMindFetchError(RuntimeError):
    def __init__(
        self,
        category,
        dataset,
        data_id,
        start_date,
        end_date,
        *,
        http_status=None,
        exception_type=None,
        blocked_until=None,
        retry_after_seconds=None,
    ):
        self.category = str(category)
        self.dataset = str(dataset)[:80]
        self.data_id = str(data_id)[:80]
        self.start_date = str(start_date)[:32]
        self.end_date = str(end_date)[:32]
        self.http_status = http_status if type(http_status) is int else None
        self.exception_type = (
            str(exception_type)[:80] if exception_type is not None else None
        )
        self.blocked_until = (
            float(blocked_until) if blocked_until is not None else None
        )
        self.retry_after_seconds = (
            int(retry_after_seconds)
            if retry_after_seconds is not None
            else None
        )
        self.provider_wide = self.category != "empty_dataset"
        status = self.http_status if self.http_status is not None else "none"
        prefix = (
            "FinMind empty dataset"
            if self.category == "empty_dataset"
            else "FinMind provider blocked"
            if self.blocked_until is not None
            else "FinMind provider failure"
        )
        self.safe_message = (
            f"{prefix}: dataset={self.dataset} status={status} "
            f"category={self.category}"
        )
        if self.blocked_until is not None:
            blocked_at = datetime.datetime.fromtimestamp(
                self.blocked_until,
                datetime.timezone.utc,
            ).isoformat().replace("+00:00", "Z")
            self.safe_message += f" blocked_until={blocked_at}"
        super().__init__(self.safe_message)

    def to_dict(self):
        return {
            "category": self.category,
            "dataset": self.dataset,
            "data_id": self.data_id,
            "start_date": self.start_date,
            "end_date": self.end_date,
            "http_status": self.http_status,
            "exception_type": self.exception_type,
            "blocked_until": self.blocked_until,
            "retry_after_seconds": self.retry_after_seconds,
            "safe_message": self.safe_message,
        }


def _finmind_failure(
    category,
    dataset,
    code,
    start_date,
    end_date,
    logger,
    **details,
):
    error = FinMindFetchError(
        category,
        dataset,
        code,
        start_date,
        end_date,
        **details,
    )
    if category == "empty_dataset":
        logger.info("%s", error.safe_message)
    else:
        logger.warning("%s", error.safe_message)
    raise error


def _retry_after_seconds(response, default):
    try:
        seconds = int((getattr(response, "headers", {}) or {}).get("Retry-After"))
    except (TypeError, ValueError):
        return default
    return seconds if 0 < seconds <= 86400 else default


def finmind_login(current_token, user, password, requests_module, *, logger):
    if current_token:
        return current_token
    if not user or not password:
        return current_token

    def fail(category, **details):
        _finmind_failure(
            category,
            "login",
            "",
            "",
            "",
            logger,
            **details,
        )

    try:
        response = requests_module.post(
            "https://api.finmindtrade.com/api/v4/login",
            data={"user_id": user, "password": password},
            timeout=5,
        )
    except requests_module.Timeout as exc:
        fail("login_timeout", exception_type=type(exc).__name__)
    except requests_module.ConnectionError as exc:
        fail("login_network_error", exception_type=type(exc).__name__)
    except requests_module.RequestException as exc:
        fail("login_network_error", exception_type=type(exc).__name__)

    status = getattr(response, "status_code", None)
    if type(status) is not int:
        fail("login_invalid_payload", exception_type="InvalidStatusCode")
    if status >= 500:
        fail(
            "login_network_error",
            http_status=status,
            exception_type="HTTPError",
        )
    if status >= 400:
        fail(
            "login_authentication",
            http_status=status,
            exception_type="HTTPError",
        )
    try:
        payload = response.json()
    except (TypeError, ValueError) as exc:
        fail("login_invalid_json", exception_type=type(exc).__name__)
    if not isinstance(payload, dict):
        fail("login_invalid_payload", exception_type="InvalidPayload")
    if payload.get("msg") != "success":
        fail("login_authentication", exception_type="AuthenticationError")
    login_token = payload.get("token")
    if not isinstance(login_token, str) or not login_token.strip():
        fail("login_invalid_payload", exception_type="InvalidToken")
    return login_token


def fetch_finmind_dataset(
    dataset,
    code,
    start_date,
    end_date,
    *,
    blocked_until,
    now,
    login,
    token,
    requests_module,
    pd,
    logger,
    sleep_fn,
    retry_attempts,
):
    timestamp = now()

    def fail(category, **details):
        _finmind_failure(
            category,
            dataset,
            code,
            start_date,
            end_date,
            logger,
            **details,
        )

    if timestamp < blocked_until:
        fail(
            "blocked",
            blocked_until=blocked_until,
        )
    login()
    params = {
        "dataset": dataset,
        "data_id": code,
        "start_date": start_date,
        "end_date": end_date,
    }
    current_token = token()
    if current_token:
        params["token"] = current_token
    retry_limit = min(max(int(retry_attempts), 0), 2)
    attempt = 0
    while True:
        try:
            response = requests_module.get(
                "https://api.finmindtrade.com/api/v4/data",
                params=params,
                timeout=8,
            )
        except requests_module.Timeout as exc:
            if attempt < retry_limit:
                sleep_fn(2**attempt)
                attempt += 1
                continue
            fail("timeout", exception_type=type(exc).__name__)
        except requests_module.ConnectionError as exc:
            if attempt < retry_limit:
                sleep_fn(2**attempt)
                attempt += 1
                continue
            fail("network_error", exception_type=type(exc).__name__)
        except requests_module.RequestException as exc:
            if attempt < retry_limit:
                sleep_fn(2**attempt)
                attempt += 1
                continue
            fail("network_error", exception_type=type(exc).__name__)

        status = getattr(response, "status_code", None)
        if type(status) is not int:
            fail("invalid_payload", exception_type="InvalidStatusCode")
        if status in (500, 502, 503, 504) and attempt < min(retry_limit, 1):
            sleep_fn(1)
            attempt += 1
            continue
        if status >= 400:
            category = {
                401: "authentication_or_permission",
                402: "quota_or_rate_limit",
                403: "authentication_or_permission",
                429: "quota_or_rate_limit",
            }.get(status, "http_error")
            retry_after = (
                _retry_after_seconds(response, 1800)
                if status == 429
                else 3600
                if status == 402
                else 1800
                if status == 403
                else None
            )
            blocked_until = (
                timestamp + retry_after if retry_after is not None else None
            )
            fail(
                category,
                http_status=status,
                exception_type="HTTPError",
                blocked_until=blocked_until,
                retry_after_seconds=retry_after,
            )
        break

    try:
        payload = response.json()
    except (TypeError, ValueError) as exc:
        fail(
            "invalid_json",
            exception_type=type(exc).__name__,
        )
    if not isinstance(payload, dict) or not isinstance(payload.get("data"), list):
        fail(
            "invalid_payload",
            exception_type="InvalidPayload",
        )
    if not payload["data"]:
        fail("empty_dataset")
    try:
        return pd.DataFrame(payload["data"]), blocked_until
    except (TypeError, ValueError) as exc:
        fail(
            "invalid_payload",
            exception_type=type(exc).__name__,
        )


def fetch_yfinance_price_history(
    tickers,
    start_date,
    end_date,
    *,
    cache,
    cache_seconds,
    now,
    pd,
    logger,
):
    if isinstance(tickers, str):
        tickers = [tickers]
    cache_key = (tuple(tickers), start_date, end_date or "")
    timestamp = now()
    cached = cache.get(cache_key)
    if cached and timestamp - cached[1] < cache_seconds:
        return cached[0].copy()

    try:
        import yfinance as yf

        for ticker in tickers:
            hist = yf.download(
                ticker,
                start=start_date,
                end=end_date,
                progress=False,
                threads=False,
            )
            if isinstance(hist.columns, pd.MultiIndex):
                hist.columns = hist.columns.droplevel(1)
            if not hist.empty and "Close" in hist.columns:
                frame = hist.copy()
                frame.index = pd.to_datetime(frame.index).tz_localize(None)
                frame.index.name = "Date"
                frame = frame.reset_index()[
                    ["Date", "Open", "High", "Low", "Close", "Volume"]
                ]
                cache[cache_key] = (frame.copy(), timestamp)
                return frame
    except Exception:
        logger.warning("Yahoo Finance 讀取失敗")
    return pd.DataFrame()


def fetch_option_context_history(
    start_date, end_date, fetch_history, executor_factory, pd, logger
):
    symbols = ("^VIX", "^VIX9D", "^VIX3M")
    frames = {}
    with executor_factory(max_workers=len(symbols)) as executor:
        futures = {
            symbol: executor.submit(fetch_history, symbol, start_date, end_date)
            for symbol in symbols
        }
        for symbol, future in futures.items():
            try:
                frames[symbol] = future.result()
            except Exception:
                logger.warning("選擇權市場指標讀取失敗 (%s)", symbol)
                frames[symbol] = pd.DataFrame()
    return tuple(frames[symbol] for symbol in symbols)


def get_stock_name(
    code,
    stock_codes,
    is_us_ticker,
    resolver=None,
    target_date=None,
    require_authoritative=False,
):
    code = str(code or "").strip().upper()
    if code == "TAIEX":
        return "台股大盤"
    if resolver is not None:
        try:
            from stock_papi.integrations.market_data.tw_security_master import (
                is_taiwan_symbol,
            )

            if is_taiwan_symbol(code):
                return resolver.resolve_name(
                    code,
                    target_date,
                    require_authoritative=require_authoritative,
                )
        except (TypeError, ValueError):
            pass
    if code in stock_codes:
        return str(getattr(stock_codes[code], "name", code))
    if is_us_ticker(code):
        return f"美股 {code}"
    return code


def search_stock_code(
    keyword,
    stock_codes,
    is_us_ticker,
    resolve_name,
    resolver=None,
):
    keyword = keyword.upper().strip()
    if not keyword:
        return None, None
    if keyword in ["TAIEX", "加權指數", "台股大盤", "大盤"]:
        return "TAIEX", "台股大盤"
    if resolver is not None:
        try:
            from stock_papi.integrations.market_data.tw_security_master import (
                is_taiwan_symbol,
            )

            if is_taiwan_symbol(keyword):
                if resolver.contains(keyword):
                    return keyword, resolve_name(keyword)
            match = resolver.search(keyword)
            if match:
                return match
        except (TypeError, ValueError):
            pass
    if keyword.isdigit():
        return keyword, resolve_name(keyword)
    if is_us_ticker(keyword):
        return keyword, resolve_name(keyword)
    for code, info in stock_codes.items():
        name = str(getattr(info, "name", "") or "")
        if keyword in name.upper():
            return code, name
    return None, None
