"""Lightweight health, search, data-freshness, and legacy redirect routes."""

import datetime as dt
from zoneinfo import ZoneInfo

from flask import jsonify, redirect, request, url_for

from stock_papi.batch.calendar import TradingCalendarSet
from stock_papi.integrations.market_data.us_calendar import (
    get_us_calendar_documents,
)
from stock_papi.integrations.market_data.tw_calendar import (
    get_tw_calendar_documents,
)


_MARKET_TIMEZONES = {
    "TW": ZoneInfo("Asia/Taipei"),
    "US": ZoneInfo("America/New_York"),
}
_UNSET = object()


def _parse_iso_date(value):
    if not isinstance(value, str):
        return None
    try:
        parsed = dt.date.fromisoformat(value)
    except ValueError:
        return None
    return parsed if parsed.isoformat() == value else None


def _parse_reference_date(value):
    if type(value) is dt.date:
        return value
    return _parse_iso_date(value)


def _utc_now():
    return dt.datetime.now(dt.timezone.utc)


def market_local_date(market, *, now):
    """Return the explicit market-local date for an aware instant."""
    timezone = _MARKET_TIMEZONES.get(market)
    if timezone is None:
        raise ValueError("unsupported market")
    if not isinstance(now, dt.datetime) or now.utcoffset() is None:
        raise ValueError("now must be timezone-aware")
    return now.astimezone(timezone).date()


def classify_data_freshness(
    *,
    source_market_date,
    applicable_trading_date,
    reference_date,
    next_session=None,
):
    """Classify verified dates against an injected market-session contract."""
    source = _parse_iso_date(source_market_date)
    applicable = _parse_iso_date(applicable_trading_date)
    reference = _parse_reference_date(reference_date)
    normalized_source = source.isoformat() if source is not None else None
    normalized_applicable = (
        applicable.isoformat() if applicable is not None else None
    )
    status = "unavailable"

    if (
        source is not None
        and applicable is not None
        and reference is not None
        and source <= applicable
        and reference >= source
    ):
        if reference <= applicable:
            status = "current"
        elif callable(next_session):
            try:
                updating_boundary = next_session(applicable)
            except Exception:
                updating_boundary = None
            if (
                type(updating_boundary) is dt.date
                and updating_boundary > applicable
            ):
                if reference < updating_boundary:
                    status = "current"
                elif reference == updating_boundary:
                    status = "updating"
                else:
                    status = "stale"

    return {
        "status": status,
        "source_market_date": normalized_source,
        "applicable_trading_date": normalized_applicable,
    }


def _latest_post_close_dates(reports, *, market):
    if not isinstance(reports, list):
        return None, None, False

    candidates = []
    for report in reports:
        if not isinstance(report, dict):
            return None, None, False
        if "market" in report and report.get("market") != market:
            return None, None, False
        if report.get("report_type") != "post_close":
            continue
        source = _parse_iso_date(report.get("source_market_date"))
        applicable = _parse_iso_date(report.get("applicable_trading_date"))
        if source is None or applicable is None or source > applicable:
            return None, None, False
        candidates.append((applicable, source))
    if not candidates:
        return None, None, True
    applicable, source = max(candidates)
    return source.isoformat(), applicable.isoformat(), True


def _next_session_for_market(market, value):
    if market == "TW":
        documents = get_tw_calendar_documents(value.year, value.year + 1)
    elif market == "US":
        documents = get_us_calendar_documents(value.year, value.year + 1)
    else:
        return None
    return TradingCalendarSet.from_documents(documents).next_session(value)


def create_data_freshness_loader(
    *,
    load_dashboard_snapshot,
    load_report_index_v2,
    now=None,
    next_session_for_market=None,
):
    """Create a lazy, market-scoped verified freshness loader."""

    def unavailable(source=None, applicable=None):
        return classify_data_freshness(
            source_market_date=source,
            applicable_trading_date=applicable,
            reference_date=None,
            next_session=None,
        )

    def load(
        market,
        *,
        reports=_UNSET,
        snapshot=_UNSET,
        reference_now=_UNSET,
    ):
        if market not in _MARKET_TIMEZONES:
            return unavailable()

        if reports is _UNSET:
            try:
                reports = load_report_index_v2(market=market)
            except Exception:
                return unavailable()
        source, applicable, valid_market = _latest_post_close_dates(
            reports, market=market
        )
        if not valid_market:
            return unavailable()

        if market == "TW":
            if snapshot is _UNSET:
                try:
                    snapshot = load_dashboard_snapshot()
                except Exception:
                    return unavailable()
            if not isinstance(snapshot, dict):
                return unavailable()
            if "market" in snapshot and snapshot.get("market") != "TW":
                return unavailable()
            dashboard_source = _parse_iso_date(snapshot.get("observation_as_of"))
            report_source = source
            source = (
                dashboard_source.isoformat()
                if dashboard_source is not None
                else None
            )
            if source is None or source != report_source:
                applicable = None

        try:
            instant = (
                (now or _utc_now)()
                if reference_now is _UNSET
                else reference_now
            )
            reference = market_local_date(market, now=instant)
        except Exception:
            return unavailable(source, applicable)

        resolver = next_session_for_market or _next_session_for_market
        return classify_data_freshness(
            source_market_date=source,
            applicable_trading_date=applicable,
            reference_date=reference,
            next_session=lambda value: resolver(market, value),
        )

    return load


def register_system_routes(
    app, *, search_stock, load_data_freshness, now=None
):
    def healthz():
        return "ok", 200

    def data_health():
        try:
            reference_now = (now or _utc_now)()
        except Exception:
            reference_now = None
        markets = {
            market: load_data_freshness(
                market, reference_now=reference_now
            )
            for market in ("TW", "US")
        }
        return jsonify({"service": {"status": "ok"}, "markets": markets})

    def search_page():
        query = request.args.get("q", "").strip()
        market = request.args.get("market", "TW").upper()
        code, _name = search_stock(query)
        if code:
            return redirect(url_for("stock_page", code=code), code=302)
        return redirect(
            url_for(
                "us_stocks_page" if market == "US" else "stocks_page",
                q=query,
                error="not-found",
            ),
            code=302,
        )

    def watchlist_page():
        return redirect("/dashboard", code=302)

    app.add_url_rule("/healthz", "healthz", healthz)
    app.add_url_rule("/health", "healthz", healthz)
    app.add_url_rule("/health/data", "data_health", data_health)
    app.add_url_rule("/search", "search_page", search_page)
    app.add_url_rule("/watchlist", "watchlist_page", watchlist_page)
