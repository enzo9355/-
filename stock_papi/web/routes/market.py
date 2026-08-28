"""Market-facing Flask route registration."""

from flask import abort, jsonify, redirect, render_template, url_for

from stock_papi.shared.formatting import safe_float as _safe_float
from stock_papi.services.model_evidence import sanitize_recommendation
from stock_papi.services.prediction_view import prediction_for


def register_market_routes(
    app, *, analyze, stock_observation, dashboard_sector_cards, cached_opportunities,
    build_market_heatmap, dashboard_top_picks, industry_map,
    market_insights_payload, twstock_codes, is_us_ticker,
    find_industry_peers, get_stock_name, dashboard_snapshot,
    us_securities_observation,
    prediction_snapshot,
):
    def dashboard_api():
        snapshot = dashboard_snapshot()
        if not isinstance(snapshot, dict):
            return jsonify(
                {
                    "status": "observation_unavailable",
                    "prediction_status": "AI 預測研究中",
                    "message": "市場觀察資料暫時無法使用",
                }
            ), 503
        if snapshot.get("product_mode") == "observation":
            return jsonify(
                {
                    **snapshot,
                    "prediction_status": "AI 預測研究中",
                }
            )

        # Verified preview candidates retain their separate research rendering.
        market = analyze("TAIEX")
        if not market:
            return jsonify({"error": "market data unavailable"}), 503
        sector_cards = dashboard_sector_cards()
        presentation = snapshot.get("presentation") or {}
        baseline_status = snapshot.get("baseline_status")
        preview_heatmap = snapshot.get("heatmap")
        preview_top_picks = snapshot.get("top_picks")
        preview_focus = snapshot.get("daily_focus")
        sectors = [
            {"name": name, "count": len(codes)}
            for name, codes in list(industry_map().items())[:8]
        ]
        return jsonify({
            "market": {
                "price": float(market["price"]), "prob": int(market["prob"]),
                "trend": market["trend"],
                "as_of": str(market.get("as_of") or ""),
                "sentiment_status": str(market.get("s_status") or "資料不足"),
                "sentiment_score": round(_safe_float(market.get("s_score")), 1),
                "confidence": str(market.get("news_confidence") or "低"),
                "recommendation": sanitize_recommendation(
                    market.get("recommendation") or {
                    "action": "控制追價",
                    "level": "insufficient",
                    "headline": "市場建議資料不足，請等待資料更新",
                    "confidence": "可信度低",
                    "supporting_reasons": [],
                    "risk_reasons": ["市場建議資料缺失"],
                    "data_as_of": str(market.get("as_of") or "") or None,
                    },
                    baseline_status,
                ),
            },
            "opportunities": cached_opportunities(),
            "sector_cards": sector_cards,
            "heatmap": (
                preview_heatmap
                if isinstance(preview_heatmap, list)
                else build_market_heatmap(sector_cards)
            ),
            "daily_focus": preview_focus if isinstance(preview_focus, list) else [],
            "top_picks": (
                preview_top_picks
                if isinstance(preview_top_picks, list)
                else dashboard_top_picks(sector_cards)
            ),
            "watchlist_hint": {
                "title": "關注與提醒在 LINE 管理",
                "steps": ["在 LINE 查詢個股", "點選加入關注", "從提醒管理設定通知"],
            },
            "sectors": sectors,
            "presentation": presentation,
            "baseline_status": baseline_status,
            "inference_as_of": snapshot.get("inference_as_of"),
            "backtest_as_of": snapshot.get("backtest_as_of"),
            "model_version": snapshot.get("model_version"),
            "backtest_version": snapshot.get("backtest_version"),
            "feature_schema_version": snapshot.get("feature_schema_version"),
            "recommendation_policy_version": snapshot.get(
                "recommendation_policy_version"
            ),
        })

    def market_insights_api():
        snapshot = dashboard_snapshot()
        if not isinstance(snapshot, dict) or snapshot.get("product_mode") != "observation":
            return jsonify({"status": "observation_unavailable"}), 503
        return jsonify(
            {
                "product_mode": "observation",
                "observation_as_of": snapshot["observation_as_of"],
                "market_observation": snapshot["market_observation"],
                "industry_observations": snapshot["industry_observations"],
                "heatmap": snapshot["heatmap"],
                "stock_events": snapshot["stock_events"],
                "trading_status_observations": snapshot.get(
                    "trading_status_observations", []
                ),
                "etf_observations": snapshot["etf_observations"],
                "data_quality": snapshot["data_quality"],
                "prediction_status": "AI 預測研究中",
            }
        )

    def market_map_page():
        return redirect(url_for("industries_page"), code=302)

    def stock_page(code):
        code = code.upper()
        if code not in twstock_codes() and not is_us_ticker(code):
            abort(404)
        data = stock_observation(code)
        market = "US" if is_us_ticker(code) else "TW"
        peer_group = find_industry_peers(code)
        peers = [{"code": peer, "name": get_stock_name(peer)} for peer in peer_group["codes"]]
        prediction = None
        if data and data.get("observation_kind") == "regular_price":
            try:
                prediction = prediction_for(
                    prediction_snapshot(market), market, code, data.get("observation_as_of")
                )
            except Exception:
                prediction = None
        return render_template(
            "stock_detail.html", d={**data, "market": market, "prediction": prediction}, peers=peers,
            peer_category=peer_group["category"],
        ) if data else "查無資料"

    def us_stocks_page():
        try:
            observation = us_securities_observation()
            if (
                not isinstance(observation, dict)
                or not isinstance(observation.get("stock_events"), list)
                or not isinstance(observation.get("etf_observations"), list)
            ):
                raise ValueError("US securities observation is invalid")
        except Exception:
            return render_template(
                "stocks.html", observation={}, search_query="",
                search_error=False, market="US", data_unavailable=True,
            ), 503
        return render_template(
            "stocks.html", observation=observation, search_query="",
            search_error=False, market="US", data_unavailable=False,
        )

    def market_page():
        snapshot = dashboard_snapshot()
        observation = (
            snapshot
            if isinstance(snapshot, dict)
            and snapshot.get("product_mode") == "observation"
            else {}
        )
        return render_template("market.html", observation=observation)

    app.add_url_rule("/api/dashboard", "dashboard_api", dashboard_api)
    app.add_url_rule("/api/market-insights", "market_insights_api", market_insights_api)
    app.add_url_rule("/market-map", "market_map_page", market_map_page)
    app.add_url_rule("/stock/<code>", "stock_page", stock_page)
    app.add_url_rule("/us/stocks", "us_stocks_page", us_stocks_page)
    app.add_url_rule("/market", "market_page", market_page)
