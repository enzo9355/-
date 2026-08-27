"""Public HTML report routes and legacy compatibility redirects."""

import copy
import datetime
import hashlib
import hmac
import json
import os
import re
import uuid

from flask import abort, make_response, redirect, render_template, url_for

from reporting.exceptions import ReportWebError
from reporting.web import find_report
from reporting.professional_html import (
    REGRESSION_ARTIFACT_UNAVAILABLE_REASON,
    build_professional_report_view,
)
from reporting.professional_binding import (
    validate_professional_report_binding,
    validate_regression_research_binding,
)
from reporting.regression_schema import (
    MAX_REGRESSION_ARTIFACT_BYTES,
    RegressionResearchArtifact,
)
from reporting.schemas import ReportMetadataV2
from stock_papi.services.report_view import build_observation_report_view
from stock_papi.services.market_summary import build_market_summary_view
from stock_papi.services.prediction_view import prediction_for
from reporting.config import MAX_CANONICAL_REPORT_BYTES
from werkzeug.exceptions import HTTPException


def _valid_report_date(report_date):
    try:
        parsed = datetime.date.fromisoformat(report_date)
    except ValueError:
        return False
    return parsed.isoformat() == report_date


def register_report_routes(
    app, *, load_index, load_metadata, load_index_v2, load_metadata_v2,
    load_metadata_v2_by_sha=None,
    load_canonical_object=None, load_regression_artifact=None,
    prediction_capability=None,
    load_data_freshness=None,
    load_prediction_snapshot=None,
):
    observation_mode = (
        prediction_capability is not None
        and prediction_capability.mode == "research"
    )

    def _secure_response(response, *, cache="public, max-age=300"):
        response.headers["Cache-Control"] = cache
        response.headers["X-Content-Type-Options"] = "nosniff"
        return response

    def _report_error(status, *, report_type=None, report_date=None, exc=None):
        correlation_id = uuid.uuid4().hex[:16]
        if exc is not None:
            app.logger.exception(
                "report_render_failed correlation_id=%s report_type=%s report_date=%s error_type=%s",
                correlation_id,
                report_type,
                report_date,
                type(exc).__name__,
            )
        response = make_response(
            render_template(
                "report_unavailable.html",
                status=status,
                correlation_id=correlation_id,
            ),
            status,
        )
        response.headers["X-Correlation-ID"] = correlation_id
        if status == 503:
            response.headers["Retry-After"] = "60"
        return _secure_response(response, cache="no-store")

    def _v2_reports(market="TW", *, required=False):
        try:
            reports = load_index_v2(market=market)
        except ReportWebError:
            raise
        except Exception as exc:
            raise ReportWebError("報告索引暫時無法使用") from exc
        if reports is None:
            if required:
                raise ReportWebError("報告索引暫時無法使用")
            return []
        if not isinstance(reports, list):
            raise ReportWebError("報告索引格式錯誤")
        if any(
            not isinstance(item, dict)
            or ("market" in item and item.get("market") != market)
            for item in reports
        ):
            raise ReportWebError("報告索引市場不一致")
        if observation_mode:
            return [
                item for item in reports
                if item.get("product_mode") == "observation"
            ]
        return reports

    def _daily_items(trading_date, market="TW"):
        reports = _v2_reports(market=market, required=True)
        return [
            item
            for item in reports
            if item.get("applicable_trading_date") == trading_date
            and item.get("report_type") in {"post_close", "pre_market"}
        ]

    def _optional_regression_overlay(*, metadata, report):
        pointer = metadata.get("regression_research")
        if pointer is None:
            return None
        try:
            metadata_obj = ReportMetadataV2.from_document(metadata)
            pointer = metadata_obj.regression_research
            if pointer is None or load_regression_artifact is None:
                raise ValueError("regression artifact loader or pointer unavailable")

            object_path = pointer["object"]
            expected_sha = pointer["sha256"]
            raw_bytes = load_regression_artifact(
                object_path,
                max_bytes=MAX_REGRESSION_ARTIFACT_BYTES,
            )
            if (
                not isinstance(raw_bytes, bytes)
                or not raw_bytes
                or len(raw_bytes) > MAX_REGRESSION_ARTIFACT_BYTES
            ):
                raise ValueError("regression artifact size invalid")
            actual_sha = hashlib.sha256(raw_bytes).hexdigest()
            if not hmac.compare_digest(actual_sha, expected_sha):
                raise ValueError("regression artifact SHA mismatch")
            if object_path != f"objects/regression/{actual_sha}.json":
                raise ValueError("regression artifact path mismatch")

            document = json.loads(raw_bytes.decode("utf-8"))
            if not isinstance(document, dict):
                raise ValueError("regression artifact must be an object")
            artifact = RegressionResearchArtifact.from_document(document)
            validate_regression_research_binding(
                metadata=metadata_obj,
                professional_report=report,
                pointer=pointer,
                regression_artifact=artifact,
            )
            return artifact
        except Exception as exc:
            app.logger.warning(
                "optional_regression_overlay_unavailable error_type=%s",
                type(exc).__name__,
            )
            return None

    def _load_verified_professional_report(*, metadata, route_source_date, expected_market):
        """Load one canonical report through the complete fail-closed contract."""
        canonical_ptr = metadata.get("professional_report")
        if not isinstance(canonical_ptr, dict):
            raise ReportWebError("報告 Canonical Object 指標遺失")

        object_path = canonical_ptr.get("object")
        expected_sha = canonical_ptr.get("sha256")
        if not isinstance(object_path, str) or not object_path:
            raise ReportWebError("報告 Canonical Object 指標遺失")
        if not isinstance(expected_sha, str) or not expected_sha:
            raise ReportWebError("報告 Canonical Object 指標遺失")
        if load_canonical_object is None:
            raise ReportWebError("系統未提供 load_canonical_object")

        try:
            raw_bytes = load_canonical_object(
                object_path, max_bytes=MAX_CANONICAL_REPORT_BYTES
            )
        except Exception as exc:
            raise ReportWebError("無法讀取 Canonical Object") from exc
        if (
            not isinstance(raw_bytes, bytes)
            or len(raw_bytes) == 0
            or len(raw_bytes) > MAX_CANONICAL_REPORT_BYTES
        ):
            raise ReportWebError("Canonical Object 內容無效")

        actual_sha256 = hashlib.sha256(raw_bytes).hexdigest()
        if not hmac.compare_digest(actual_sha256, expected_sha):
            raise ReportWebError("Canonical Object 雜湊比對失敗")
        if object_path != f"objects/canonical/{actual_sha256}.json":
            raise ReportWebError("Canonical Object 路徑與雜湊不符")

        try:
            canonical_doc = json.loads(raw_bytes.decode("utf-8"))
        except UnicodeDecodeError as exc:
            raise ReportWebError("Canonical Object 解碼失敗") from exc
        except json.JSONDecodeError as exc:
            raise ReportWebError("Canonical Object JSON 解析失敗") from exc
        if not isinstance(canonical_doc, dict):
            raise ReportWebError("Canonical Object 格式錯誤")

        from reporting.professional_schema import ProfessionalPostCloseReport
        try:
            report = ProfessionalPostCloseReport.from_document(canonical_doc)
        except (ValueError, TypeError, KeyError) as exc:
            raise ReportWebError("Canonical Object 驗證失敗") from exc

        if metadata.get("market") != expected_market or report.identity.market != expected_market:
            raise ReportWebError("Professional Report 市場不符")

        try:
            canonical_metadata = copy.deepcopy(metadata)
            canonical_metadata.pop("regression_research", None)
            validate_professional_report_binding(
                route_source_date=route_source_date,
                metadata=canonical_metadata,
                pointer=canonical_ptr,
                report=report,
            )
        except (ValueError, TypeError) as exc:
            raise ReportWebError("Professional Report 綁定驗證失敗") from exc
        return report

    def _observation_page(date_param: str, report_type: str, market="TW"):
        try:
            reports = _v2_reports(market=market, required=True)
            if report_type == "post_close":
                # For post_close, canonical date_param is source_market_date
                item = next(
                    (value for value in reports
                     if value.get("report_type") == report_type
                     and value.get("source_market_date") == date_param),
                    None,
                )
                if item is None:
                    # Fallback: if navigated with applicable_trading_date, redirect to canonical source_market_date URL
                    fallback = next(
                        (value for value in reports
                         if value.get("report_type") == report_type
                         and value.get("applicable_trading_date") == date_param),
                        None,
                    )
                    if fallback is not None:
                        canonical_date = fallback.get("source_market_date")
                        if canonical_date and canonical_date != date_param:
                            endpoint = "post_close_report_page" if market == "TW" else "us_post_close_report_page"
                            return redirect(url_for(endpoint, trading_date=canonical_date), code=302)
            else:
                # For pre_market, date_param is applicable_trading_date
                item = next(
                    (value for value in reports
                     if value.get("report_type") == report_type
                     and value.get("applicable_trading_date") == date_param),
                    None,
                )

            if item is None:
                abort(404)
            metadata = load_metadata_v2(item, expected_market=market)
            if metadata is None:
                raise ReportWebError("報告內容暫時無法使用")
            
            if report_type == "pre_market":
                content = metadata.get("content")
                base_metadata_sha256 = (
                    content.get("base_metadata_sha256")
                    if isinstance(content, dict)
                    else None
                )
                base_metadata = (
                    load_metadata_v2_by_sha(
                        base_metadata_sha256, expected_market=market
                    )
                    if load_metadata_v2_by_sha is not None
                    else None
                )
                if base_metadata is not None:
                    if (
                        base_metadata.get("report_type") != "post_close"
                        or base_metadata.get("market", "TW") != market
                        or base_metadata.get("source_market_date")
                        != metadata.get("source_market_date")
                        or base_metadata.get("applicable_trading_date")
                        != metadata.get("applicable_trading_date")
                    ):
                        raise ReportWebError("盤前報告盤後基底不一致")
                    expected_base_metadata_sha256 = base_metadata_sha256
                else:
                    post_close_item = next(
                        (
                            value for value in reports
                            if value.get("report_type") == "post_close"
                            and value.get("applicable_trading_date") == date_param
                        ),
                        None,
                    )
                    if post_close_item is None:
                        raise ReportWebError("盤前報告缺少盤後基底")
                    expected_base_metadata_sha256 = post_close_item.get(
                        "metadata_sha256"
                    )
                report = build_observation_report_view(
                    metadata,
                    expected_base_metadata_sha256=expected_base_metadata_sha256,
                )
                response = make_response(
                    render_template("report_observation.html", report=report)
                )
            elif report_type == "post_close":
                canonical_ptr = metadata.get("professional_report")
                if canonical_ptr is None and metadata.get("product_mode") == "observation":
                    report = build_observation_report_view(metadata)
                    response = make_response(
                        render_template("report_observation.html", report=report)
                    )
                    return _secure_response(response)
                prof_report = _load_verified_professional_report(
                    metadata=metadata,
                    route_source_date=date_param,
                    expected_market=market,
                )

                pdf_download_url = None
                regression_artifact = _optional_regression_overlay(
                    metadata=metadata,
                    report=prof_report,
                )
                view_model = build_professional_report_view(
                    prof_report,
                    regression_artifact=regression_artifact,
                    regression_unavailable_reason=REGRESSION_ARTIFACT_UNAVAILABLE_REASON,
                    pdf_download_url=pdf_download_url,
                )
                response = make_response(
                    render_template("reports/post_close_professional.html", report=view_model)
                )
            else:
                abort(404)
            return _secure_response(response)
        except ReportWebError as exc:
            return _report_error(
                503,
                report_type=report_type,
                report_date=date_param,
                exc=exc,
            )
        except HTTPException:
            raise
        except Exception as exc:
            return _report_error(
                500,
                report_type=report_type,
                report_date=date_param,
                exc=exc,
            )

    def reports_page():
        if observation_mode:
            reports = []
        else:
            try:
                reports = load_index()
            except ReportWebError:
                reports = None
        try:
            reports_v2 = _v2_reports(required=observation_mode)
        except ReportWebError as exc:
            if observation_mode:
                return _report_error(503, exc=exc)
            reports_v2 = None
        response = make_response(render_template(
            "reports.html", reports=reports or [], reports_v2=reports_v2 or [],
            unavailable=reports is None and reports_v2 is None,
            market="TW",
        ))
        return _secure_response(response)

    def report_page(report_date):
        if observation_mode:
            abort(404)
        if not _valid_report_date(report_date):
            abort(404)
        try:
            reports = load_index()
        except ReportWebError:
            return "報告服務暫時無法使用", 503
        if reports is None:
            return "報告服務暫時無法使用", 503
        item = find_report(reports, report_date)
        if item is None:
            abort(404)
        try:
            metadata = load_metadata(item)
        except ReportWebError:
            return "報告內容暫時無法使用", 503
        if metadata is None:
            return "報告內容暫時無法使用", 503
        response = make_response(render_template(
            "report_detail.html",
            report=item,
            metadata=metadata,
            public_report=metadata.get("public_report"),
        ))
        return _secure_response(response)

    def legacy_report_redirect(report_date):
        if not _valid_report_date(report_date):
            abort(404)
        return redirect(url_for("report_page", report_date=report_date), code=302)

    def sample_report_download():
        return redirect(url_for("reports_page"), code=302)

    def trading_day_report_page(trading_date):
        if not _valid_report_date(trading_date):
            abort(404)
        try:
            items = _daily_items(trading_date)
            if not items:
                abort(404)
            response = make_response(render_template(
                "report_day_index.html",
                trading_date=trading_date,
                reports=items,
            ))
            return _secure_response(response)
        except ReportWebError as exc:
            return _report_error(503, report_date=trading_date, exc=exc)
        except HTTPException:
            raise
        except Exception as exc:
            return _report_error(500, report_date=trading_date, exc=exc)

    def post_close_report_page(trading_date):
        if not _valid_report_date(trading_date):
            abort(404)
        return _observation_page(trading_date, "post_close", market="TW")

    def pre_market_report_page(trading_date):
        if not _valid_report_date(trading_date):
            abort(404)
        return _observation_page(trading_date, "pre_market", market="TW")

    def us_reports_page():
        try:
            reports_v2 = _v2_reports(market="US", required=observation_mode)
        except ReportWebError as exc:
            if observation_mode:
                return _report_error(503, exc=exc)
            reports_v2 = None
        response = make_response(render_template(
            "reports.html", reports=[], reports_v2=reports_v2 or [],
            unavailable=reports_v2 is None,
            market="US",
        ))
        return _secure_response(response)

    def _us_summary_response(template_name):
        try:
            reports = _v2_reports(market="US", required=True)
            item = next(
                (value for value in reports if value.get("report_type") == "post_close"),
                None,
            )
            if item is None:
                raise ReportWebError("美股盤後報告暫時無法使用")
            metadata = load_metadata_v2(item, expected_market="US")
            if metadata is None:
                raise ReportWebError("美股報告內容暫時無法使用")
            report = _load_verified_professional_report(
                metadata=metadata,
                route_source_date=item.get("source_market_date"),
                expected_market="US",
            )
            if report.identity.market != "US":
                raise ReportWebError("美股 Canonical Object 市場不一致")
            summary = build_market_summary_view(report)
            context = {
                "summary": summary,
                "market": "US",
            }
            if template_name == "us_dashboard.html" and load_prediction_snapshot:
                predictions = []
                try:
                    product = load_prediction_snapshot("US")
                    for symbol, name in (
                        ("^GSPC", "S&P 500"),
                        ("^IXIC", "Nasdaq Composite"),
                        ("^DJI", "道瓊工業指數"),
                    ):
                        value = prediction_for(
                            product, "US", symbol, summary["source_market_date"]
                        )
                        if value is not None:
                            predictions.append({**value, "symbol": symbol, "name": name})
                except Exception:
                    predictions = []
                context["index_predictions"] = predictions
            if template_name == "us_dashboard.html" and load_data_freshness:
                try:
                    context["data_freshness"] = {
                        "US": load_data_freshness("US", reports=reports)
                    }
                except Exception:
                    context["data_freshness"] = {}
            response = make_response(render_template(template_name, **context))
            return _secure_response(response)
        except ReportWebError as exc:
            return _report_error(503, report_type="post_close", exc=exc)
        except Exception as exc:
            return _report_error(500, report_type="post_close", exc=exc)

    def us_dashboard_page():
        return _us_summary_response("us_dashboard.html")

    def us_market_page():
        return _us_summary_response("us_market.html")

    def us_industries_page():
        return _us_summary_response("us_industries.html")

    def us_post_close_report_page(trading_date):
        if not _valid_report_date(trading_date):
            abort(404)
        return _observation_page(trading_date, "post_close", market="US")

    def us_pre_market_report_page(trading_date):
        if not _valid_report_date(trading_date):
            abort(404)
        return _observation_page(trading_date, "pre_market", market="US")

    def us_trading_day_report_page(trading_date):
        if not _valid_report_date(trading_date):
            abort(404)
        try:
            items = _daily_items(trading_date, market="US")
            if not items:
                abort(404)
            if any(item.get("report_type") == "post_close" for item in items):
                post_close_item = next(item for item in items if item.get("report_type") == "post_close")
                canonical_source_date = post_close_item.get("source_market_date") or trading_date
                return redirect(
                    url_for("us_post_close_report_page", trading_date=canonical_source_date),
                    code=302,
                )
            if any(item.get("report_type") == "pre_market" for item in items):
                return redirect(
                    url_for("us_pre_market_report_page", trading_date=trading_date),
                    code=302,
                )
            abort(404)
        except ReportWebError as exc:
            return _report_error(503, report_date=trading_date, exc=exc)
        except HTTPException:
            raise
        except Exception as exc:
            return _report_error(500, report_date=trading_date, exc=exc)

    def weekly_report_page(week_id):
        if observation_mode:
            abort(404)
        if not isinstance(week_id, str) or re.fullmatch(r"[0-9]{4}-W[0-9]{2}", week_id) is None:
            abort(404)
        return _report_error(503)

    app.add_url_rule("/reports", "reports_page", reports_page)
    app.add_url_rule("/reports/<report_date>", "report_page", report_page)
    app.add_url_rule(
        "/reports/<report_date>/preview", "report_preview", legacy_report_redirect
    )
    app.add_url_rule(
        "/reports/<report_date>/download", "report_download", legacy_report_redirect
    )
    app.add_url_rule(
        "/reports/sample/download", "sample_report_download", sample_report_download
    )
    app.add_url_rule(
        "/reports/trading-day/<trading_date>",
        "trading_day_report_page",
        trading_day_report_page,
    )
    app.add_url_rule(
        "/reports/<trading_date>/post-close",
        "post_close_report_page",
        post_close_report_page,
    )
    app.add_url_rule(
        "/reports/<trading_date>/pre-market",
        "pre_market_report_page",
        pre_market_report_page,
    )
    app.add_url_rule(
        "/reports/us",
        "us_reports_page",
        us_reports_page,
    )
    app.add_url_rule("/us", "us_dashboard_page", us_dashboard_page)
    app.add_url_rule("/us/market", "us_market_page", us_market_page)
    app.add_url_rule("/us/industries", "us_industries_page", us_industries_page)
    app.add_url_rule(
        "/reports/us/<trading_date>/post-close",
        "us_post_close_report_page",
        us_post_close_report_page,
    )
    app.add_url_rule(
        "/reports/us/<trading_date>/pre-market",
        "us_pre_market_report_page",
        us_pre_market_report_page,
    )
    app.add_url_rule(
        "/reports/us/trading-day/<trading_date>",
        "us_trading_day_report_page",
        us_trading_day_report_page,
    )
    app.add_url_rule(
        "/reports/weekly/<week_id>", "weekly_report_page", weekly_report_page
    )
