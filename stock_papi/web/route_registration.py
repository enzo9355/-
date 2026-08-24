"""Central registration for the public Flask route inventory."""

from stock_papi.web.routes.dashboard import register_dashboard_page
from stock_papi.web.routes.market import register_market_routes
from stock_papi.web.routes.reports import register_report_routes
from stock_papi.web.routes.system import register_system_routes
from stock_papi.web.routes.auth import register_auth_routes
from stock_papi.integrations.line.webhook import register_line_routes
from absorb.conversation.web import register_conversation_routes


def register_routes(app, dependencies):
    register_dashboard_page(
        app,
        load_report_index_v2=dependencies["load_report_index_v2"],
        load_dashboard_snapshot=dependencies["dashboard_snapshot"],
        preview_enabled=(
            dependencies["prediction_capability"].preview_candidate_prefix
            is not None
        ),
    )
    register_system_routes(
        app,
        search_stock=dependencies["search_stock"],
        load_dashboard_snapshot=dependencies["dashboard_snapshot"],
        load_report_index_v2=dependencies["load_report_index_v2"],
    )
    register_report_routes(
        app,
        load_index=dependencies["load_report_index"],
        load_metadata=dependencies["load_report_metadata"],
        load_index_v2=dependencies["load_report_index_v2"],
        load_metadata_v2=dependencies["load_report_metadata_v2"],
        load_metadata_v2_by_sha=dependencies["load_report_metadata_v2_by_sha"],
        load_canonical_object=dependencies["load_canonical_object"],
        load_regression_artifact=dependencies["load_regression_artifact"],
        prediction_capability=dependencies["prediction_capability"],
    )
    register_market_routes(
        app,
        analyze=dependencies["analyze"],
        stock_observation=dependencies["stock_observation"],
        dashboard_sector_cards=dependencies["dashboard_sector_cards"],
        cached_opportunities=dependencies["cached_opportunities"],
        build_market_heatmap=dependencies["build_market_heatmap"],
        dashboard_top_picks=dependencies["dashboard_top_picks"],
        industry_map=dependencies["industry_map"],
        market_insights_payload=dependencies["market_insights_payload"],
        twstock_codes=dependencies["twstock_codes"],
        is_us_ticker=dependencies["is_us_ticker"],
        find_industry_peers=dependencies["find_industry_peers"],
        get_stock_name=dependencies["get_stock_name"],
        dashboard_snapshot=dependencies["dashboard_snapshot"],
    )
    register_auth_routes(
        app,
        config=dependencies["line_login_config"],
        auth_store=dependencies["get_auth_store"],
        line_store=dependencies["get_line_store"],
        search_stock=dependencies["search_stock"],
        http_post=dependencies["auth_http_post"],
        now=dependencies["auth_now"],
    )
    register_conversation_routes(
        app,
        converse=dependencies["converse"],
        resolve_authenticated_identity=dependencies["resolve_conversation_identity"],
    )
    register_line_routes(
        app,
        handler=dependencies["handler"],
        get_line_bot_api=dependencies["get_line_bot_api"],
        get_line_store=dependencies["get_line_store"],
        get_broadcast_token=dependencies["get_broadcast_token"],
        get_alert_task_token=dependencies["get_alert_task_token"],
        analyze=dependencies["analyze"],
        observe=dependencies["stock_observation"],
        observation_mode=(
            dependencies["prediction_capability"].mode == "research"
        ),
        get_broadcast_insight=dependencies["get_broadcast_insight"],
        refresh_sector_signals=dependencies["refresh_sector_signals"],
        run_alert_checks=dependencies["run_alert_checks"],
    )
