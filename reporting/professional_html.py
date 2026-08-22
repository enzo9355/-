"""Safe HTML view-model adapter for canonical professional reports."""

from __future__ import annotations

import copy
from typing import Any

from .professional_schema import ProfessionalPostCloseReport, ProfessionalSection
from .regression_schema import RegressionResearchArtifact


REGRESSION_UNAVAILABLE_REASON = "量化回歸研究目前無法安全顯示。"
REGRESSION_ARTIFACT_UNAVAILABLE_REASON = "量化回歸研究尚未提供。"
REGRESSION_UNAVAILABLE_REASONS = frozenset(
    {
        REGRESSION_UNAVAILABLE_REASON,
        REGRESSION_ARTIFACT_UNAVAILABLE_REASON,
    }
)


def _section_view(section: ProfessionalSection) -> dict[str, Any]:
    return {
        "status": section.status,
        "data_as_of": section.data_as_of.isoformat() if section.data_as_of else None,
        "reason": section.reason,
        "data": copy.deepcopy(section.data),
    }


def build_professional_report_view(
    report: ProfessionalPostCloseReport,
    *,
    regression_artifact: RegressionResearchArtifact | None = None,
    regression_unavailable_reason: str | None = None,
    pdf_download_url: str | None = None,
) -> dict[str, Any]:
    """Return a Jinja-safe view model without internal object paths or raw metadata."""

    if not isinstance(report, ProfessionalPostCloseReport):
        raise TypeError("report must be ProfessionalPostCloseReport")
    if regression_artifact is not None and not isinstance(
        regression_artifact, RegressionResearchArtifact
    ):
        raise TypeError("regression_artifact must be RegressionResearchArtifact")

    identity = report.identity
    if regression_artifact is None:
        safe_reason = (
            regression_unavailable_reason
            if isinstance(regression_unavailable_reason, str)
            and regression_unavailable_reason in REGRESSION_UNAVAILABLE_REASONS
            else REGRESSION_UNAVAILABLE_REASON
        )
        quantitative_research = {
            "status": "unavailable",
            "reason": safe_reason,
            "data": {},
        }
    else:
        spec = regression_artifact.regression_spec
        diagnostics = regression_artifact.diagnostics
        presentation = regression_artifact.presentation
        quantitative_research = {
            "status": "available",
            "reason": None,
            "data": {},
            "ai_label": "AI 模型參考建議",
            "output_name": "模型方向參考",
            "headline": presentation.headline,
            "summary": presentation.summary,
            "sample_count": spec.sample_count,
            "first_feature_session": spec.first_feature_session,
            "last_feature_session": spec.last_feature_session,
            "dependent_variable": spec.dependent_variable,
            "covariance_estimator": spec.covariance_estimator,
            "hac_max_lags": spec.hac_max_lags,
            "confidence_level": spec.confidence_level,
            "confidence_level_pct": int(round(spec.confidence_level * 100)),
            "factors": [item.to_dict() for item in regression_artifact.results],
            "fit_statistics": regression_artifact.fit_statistics.to_dict(),
            "r_squared": regression_artifact.fit_statistics.r_squared,
            "adjusted_r_squared": regression_artifact.fit_statistics.adjusted_r_squared,
            "diagnostics": diagnostics.to_dict(),
            "warnings": copy.deepcopy(diagnostics.warnings),
            "key_exposures": copy.deepcopy(presentation.key_exposures),
            "limitations": presentation.limitations,
            "disclosure": presentation.disclosure,
        }
    capital_flows_view = _section_view(report.capital_flows)
    if capital_flows_view.get("status") == "available" and isinstance(
        capital_flows_view.get("data"), dict
    ):
        flows = capital_flows_view["data"]
        valid_items = [
            flows[k]
            for k in ("foreign_net", "investment_trust_net", "dealer_net")
            if isinstance(flows.get(k), (int, float))
        ]
        if valid_items:
            flows["total_institutional_net"] = sum(valid_items)

    return {
        "title": (
            "ABSORB "
            f"{'美股' if identity.market == 'US' else '台股'}市場、產業與量化研究日報"
        ),
        "identity": {
            "report_id": identity.report_id,
            "market": identity.market,
            "source_market_date": identity.source_market_date.isoformat(),
            "applicable_trading_date": identity.applicable_trading_date.isoformat(),
            "published_at": identity.published_at.isoformat(),
            "content_sha256_short": identity.content_sha256[:12],
            "source_manifest_sha256_short": identity.source_manifest_sha256[:12],
            "code_commit_sha_short": identity.code_commit_sha[:12],
            "generator_version": identity.generator_version,
            "feature_schema_version": identity.feature_schema_version,
            "recommendation_policy_version": identity.recommendation_policy_version,
            "model_version": identity.model_version,
        },
        "executive_summary": report.executive_summary.to_document(),
        "key_events": copy.deepcopy(list(report.key_events)),
        "market": _section_view(report.market),
        "capital_flows": capital_flows_view,
        "industries": _section_view(report.industries),
        "securities": _section_view(report.securities),
        "quantitative_research": quantitative_research,
        "validation": _section_view(report.validation),
        "next_session": _section_view(report.next_session),
        "governance": _section_view(report.governance),
        "ai_reference": _section_view(report.ai_reference),
        "pdf_download_url": pdf_download_url,
    }
