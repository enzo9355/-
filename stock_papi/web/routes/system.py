"""Lightweight health, search, data-freshness, and legacy redirect routes."""

import datetime as dt

from flask import g, jsonify, redirect, request, url_for


def _parse_iso_date(value):
    if not isinstance(value, str):
        return None
    try:
        parsed = dt.date.fromisoformat(value)
    except ValueError:
        return None
    return parsed if parsed.isoformat() == value else None


def classify_data_freshness(
    *, source_market_date, applicable_trading_date, reference_date
):
    """Classify verified market dates without inventing a trading session."""
    source = _parse_iso_date(source_market_date)
    applicable = _parse_iso_date(applicable_trading_date)
    reference = _parse_iso_date(reference_date)
    status = "unavailable"
    if source is not None and applicable is not None and reference is not None:
        if source == reference and applicable == reference:
            status = "current"
        elif source < reference <= applicable:
            status = "updating"
        elif source < reference and applicable < reference:
            status = "stale"
    return {
        "status": status,
        "source_market_date": source_market_date,
        "applicable_trading_date": applicable_trading_date,
    }


def _latest_post_close_dates(reports):
    candidates = []
    for report in reports if isinstance(reports, list) else ():
        if not isinstance(report, dict) or report.get("report_type") != "post_close":
            continue
        source = report.get("source_market_date")
        applicable = report.get("applicable_trading_date")
        if _parse_iso_date(source) is not None and _parse_iso_date(applicable) is not None:
            candidates.append((applicable, source))
    if not candidates:
        return None, None
    applicable, source = max(candidates)
    return source, applicable


def register_system_routes(
    app, *, search_stock, load_dashboard_snapshot, load_report_index_v2
):
    def _load_reports(market):
        try:
            reports = load_report_index_v2(market=market)
        except TypeError:
            reports = load_report_index_v2()
        except Exception:
            return []
        return reports if isinstance(reports, list) else []

    def _data_freshness():
        cached = getattr(g, "_data_freshness", None)
        if cached is not None:
            return cached
        try:
            snapshot = load_dashboard_snapshot()
        except Exception:
            snapshot = None
        tw_source = (
            snapshot.get("observation_as_of") if isinstance(snapshot, dict) else None
        )
        tw_report_source, tw_applicable = _latest_post_close_dates(
            _load_reports("TW")
        )
        if tw_report_source != tw_source:
            tw_applicable = None
        us_source, us_applicable = _latest_post_close_dates(_load_reports("US"))
        reference_date = dt.date.today().isoformat()
        g._data_freshness = {
            "TW": classify_data_freshness(
                source_market_date=tw_source,
                applicable_trading_date=tw_applicable,
                reference_date=reference_date,
            ),
            "US": classify_data_freshness(
                source_market_date=us_source,
                applicable_trading_date=us_applicable,
                reference_date=reference_date,
            ),
        }
        return g._data_freshness

    def healthz():
        return "ok", 200

    def data_health():
        return jsonify({"service": {"status": "ok"}, "markets": _data_freshness()})

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

    @app.context_processor
    def data_freshness_context():
        return {"data_freshness": _data_freshness()}

    app.add_url_rule("/healthz", "healthz", healthz)
    app.add_url_rule("/health", "healthz", healthz)
    app.add_url_rule("/health/data", "data_health", data_health)
    app.add_url_rule("/search", "search_page", search_page)
    app.add_url_rule("/watchlist", "watchlist_page", watchlist_page)
