"""Lightweight health, search, and legacy redirect routes."""

from flask import redirect, request, url_for


def register_system_routes(app, *, search_stock):
    def healthz():
        return "ok", 200

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
    app.add_url_rule("/search", "search_page", search_page)
    app.add_url_rule("/watchlist", "watchlist_page", watchlist_page)
