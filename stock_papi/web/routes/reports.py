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
    load_canonical_object=None, load_regression_artifact=None,
    prediction_capability=None,
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

    def _v2_reports(*, required=False):
        reports = load_index_v2()
        if reports is None:
            if required:
                raise ReportWebError("報告索引暫時無法使用")
            return []
        if observation_mode:
            return [
                item for item in reports
                if item.get("product_mode") == "observation"
            ]
        return reports

    def _daily_items(trading_date):
        reports = _v2_reports(required=True)
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

    def _observation_page(date_param: str, report_type: str):
        try:
            reports = _v2_reports(required=True)
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
                            return redirect(url_for("post_close_report_page", trading_date=canonical_date), code=302)
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
            metadata = load_metadata_v2(item)
            if metadata is None:
                raise ReportWebError("報告內容暫時無法使用")
            
            if report_type == "pre_market":
                expected_base_metadata_sha256 = None
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

                expected_object = f"objects/canonical/{actual_sha256}.json"
                if object_path != expected_object:
                    raise ReportWebError("Canonical Object 路徑與雜湊不符")

                try:
                    text_content = raw_bytes.decode("utf-8")
                except UnicodeDecodeError as exc:
                    raise ReportWebError("Canonical Object 解碼失敗") from exc

                try:
                    canonical_doc = json.loads(text_content)
                except json.JSONDecodeError as exc:
                    raise ReportWebError("Canonical Object JSON 解析失敗") from exc

                if not isinstance(canonical_doc, dict):
                    raise ReportWebError("Canonical Object 格式錯誤")

                from reporting.professional_schema import ProfessionalPostCloseReport
                try:
                    prof_report = ProfessionalPostCloseReport.from_document(canonical_doc)
                except (ValueError, TypeError, KeyError) as exc:
                    raise ReportWebError("Canonical Object 驗證失敗") from exc

                try:
                    canonical_metadata = copy.deepcopy(metadata)
                    canonical_metadata.pop("regression_research", None)
                    validate_professional_report_binding(
                        route_source_date=date_param,
                        metadata=canonical_metadata,
                        pointer=canonical_ptr,
                        report=prof_report,
                    )
                except (ValueError, TypeError) as exc:
                    raise ReportWebError("Professional Report 綁定驗證失敗") from exc

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
            unavailable=reports is None and reports_v2 is None
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
        return _observation_page(trading_date, "post_close")

    def pre_market_report_page(trading_date):
        if not _valid_report_date(trading_date):
            abort(404)
        return _observation_page(trading_date, "pre_market")

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
        "/reports/weekly/<week_id>", "weekly_report_page", weekly_report_page
    )
