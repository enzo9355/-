"""Dashboard page route registration."""

from flask import abort, render_template, request

from reporting.exceptions import ReportWebError


def register_dashboard_page(
    app,
    *,
    load_report_index_v2,
    load_dashboard_snapshot,
    load_data_freshness=None,
    preview_enabled=False,
):
    def _snapshot():
        value = load_dashboard_snapshot() or {}
        return value if isinstance(value, dict) else {}

    def dashboard_page():
        try:
            reports = load_report_index_v2(market="TW") or []
        except ReportWebError:
            reports = []
        reports = [
            item
            for item in reports
            if isinstance(item, dict) and item.get("market", "TW") == "TW"
        ]
        daily_cards = {
            report_type: next(
                (item for item in reports if item.get("report_type") == report_type),
                None,
            )
            for report_type in ("post_close", "pre_market")
        }
        snapshot = _snapshot()
        data_freshness = {}
        if load_data_freshness is not None:
            try:
                data_freshness["TW"] = load_data_freshness(
                    "TW", reports=reports, snapshot=snapshot
                )
            except Exception:
                pass
        return render_template(
            "dashboard.html",
            search_query=request.args.get("q", "").strip(),
            search_error=request.args.get("error") == "not-found",
            daily_cards=daily_cards,
            model_presentation=snapshot.get("presentation") or {},
            observation=snapshot,
            data_freshness=data_freshness,
        )

    def industries_page():
        return render_template("industries.html", observation=_snapshot())

    def stocks_page():
        return render_template(
            "stocks.html",
            observation=_snapshot(),
            search_query=request.args.get("q", "").strip(),
            search_error=request.args.get("error") == "not-found",
        )

    def ask_page():
        return render_template("ask.html")

    def learn_page():
        return render_template("learn.html")

    def preview_report_page():
        if not preview_enabled:
            abort(404)
        snapshot = load_dashboard_snapshot()
        if not isinstance(snapshot, dict):
            abort(404)
        return render_template("preview_report.html", snapshot=snapshot)

    app.add_url_rule("/", "dashboard_page", dashboard_page)
    app.add_url_rule("/dashboard", "dashboard_page", dashboard_page)
    app.add_url_rule("/industries", "industries_page", industries_page)
    app.add_url_rule("/stocks", "stocks_page", stocks_page)
    app.add_url_rule("/ask", "ask_page", ask_page)
    app.add_url_rule("/learn", "learn_page", learn_page)
    app.add_url_rule("/preview/report", "preview_report_page", preview_report_page)
