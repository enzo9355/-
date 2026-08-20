"""Validation module for binding professional post-close report with metadata and routes.
"""

from __future__ import annotations

import datetime as dt
import re
from typing import Any

from .professional_schema import ProfessionalPostCloseReport, compute_content_sha256
from .regression_schema import RegressionResearchArtifact
from .schemas import ReportMetadataV2


def validate_professional_report_binding(
    *,
    route_source_date: dt.date | str | None = None,
    metadata: ReportMetadataV2 | dict[str, Any],
    pointer: dict[str, Any] | None = None,
    report: ProfessionalPostCloseReport | dict[str, Any],
) -> None:
    """Cross-check professional post-close report against metadata, pointer, and optional route parameters."""

    if isinstance(report, ProfessionalPostCloseReport):
        report_obj = report
        report_doc = report.to_document()
    elif isinstance(report, dict):
        report_obj = ProfessionalPostCloseReport.from_document(report)
        report_doc = dict(report)
    else:
        raise ValueError("report must be a ProfessionalPostCloseReport instance or document dict")

    identity = report_obj.identity

    if isinstance(metadata, ReportMetadataV2):
        meta_obj = metadata
    elif isinstance(metadata, dict):
        meta_obj = ReportMetadataV2.from_document(metadata)
    else:
        raise ValueError("metadata must be a ReportMetadataV2 instance or document dict")

    # 1. Identity static field constraints
    if identity.report_type != "post_close":
        raise ValueError("report_type must be post_close")
    if identity.market not in ("TW", "US"):
        raise ValueError("market must be TW or US")
    if identity.product_mode != "observation_with_research":
        raise ValueError("product_mode must be observation_with_research")
    if identity.product_tier != "institutional":
        raise ValueError("product_tier must be institutional")

    expected_report_id = f"{identity.market}-{identity.source_market_date.strftime('%Y%m%d')}-post-close-institutional"
    if identity.report_id != expected_report_id:
        raise ValueError(f"report_id '{identity.report_id}' does not match expected '{expected_report_id}'")
    # 2. Metadata cross-checks
    if meta_obj.report_type != "post_close":
        raise ValueError("metadata report_type must be post_close")
    if meta_obj.market not in ("TW", "US"):
        raise ValueError("metadata market must be TW or US")
    if identity.market != meta_obj.market:
        raise ValueError("market mismatch between report identity and metadata")
    if identity.source_market_date != meta_obj.source_market_date:
        raise ValueError("source_market_date mismatch between report identity and metadata")
    if identity.applicable_trading_date != meta_obj.applicable_trading_date:
        raise ValueError("applicable_trading_date mismatch between report identity and metadata")
    if identity.source_manifest != meta_obj.source_manifest:
        raise ValueError("source_manifest mismatch between report identity and metadata")
    if identity.source_manifest_sha256 != meta_obj.source_manifest_sha256:
        raise ValueError("source_manifest_sha256 mismatch between report identity and metadata")

    # 3. Route source date cross-check
    if route_source_date is not None:
        if isinstance(route_source_date, str):
            parsed_route_date = dt.date.fromisoformat(route_source_date)
        elif isinstance(route_source_date, dt.date):
            parsed_route_date = route_source_date
        else:
            raise ValueError("route_source_date must be a date or ISO string")

        if identity.source_market_date != parsed_route_date:
            raise ValueError("route_source_date does not match report source_market_date")

    # 4. Content SHA255 re-calculation check
    recalculated_sha = compute_content_sha256(report_doc)
    if identity.content_sha256 != recalculated_sha:
        raise ValueError("content_sha256 does not match recalculated content hash")

    # 5. Pointer cross-checks
    pointer_to_check = pointer if pointer is not None else meta_obj.professional_report
    if pointer_to_check is not None:
        if isinstance(pointer_to_check, dict):
            ptr_content_sha = pointer_to_check.get("content_sha256")
            ptr_gen_ver = pointer_to_check.get("generator_version")
            ptr_commit_sha = pointer_to_check.get("code_commit_sha")
        else:
            ptr_content_sha = getattr(pointer_to_check, "content_sha256", None)
            ptr_gen_ver = getattr(pointer_to_check, "generator_version", None)
            ptr_commit_sha = getattr(pointer_to_check, "code_commit_sha", None)

        if ptr_content_sha != identity.content_sha256:
            raise ValueError("pointer content_sha256 does not match report identity content_sha256")
        if ptr_gen_ver is not None and ptr_gen_ver != identity.generator_version:
            raise ValueError("pointer generator_version does not match report identity generator_version")
        if ptr_commit_sha is not None and ptr_commit_sha != identity.code_commit_sha:
            raise ValueError("pointer code_commit_sha does not match report identity code_commit_sha")


def validate_regression_research_binding(
    *,
    metadata: ReportMetadataV2 | dict[str, Any],
    professional_report: ProfessionalPostCloseReport | dict[str, Any],
    pointer: dict[str, Any],
    regression_artifact: RegressionResearchArtifact | dict[str, Any],
) -> None:
    """Bind an optional regression artifact to metadata and canonical report lineage."""
    meta_obj = (
        metadata
        if isinstance(metadata, ReportMetadataV2)
        else ReportMetadataV2.from_document(metadata)
    )
    report_obj = (
        professional_report
        if isinstance(professional_report, ProfessionalPostCloseReport)
        else ProfessionalPostCloseReport.from_document(professional_report)
    )
    if isinstance(regression_artifact, RegressionResearchArtifact):
        artifact_obj = regression_artifact
    else:
        artifact_obj = RegressionResearchArtifact.from_document(regression_artifact)
    if not isinstance(pointer, dict):
        raise ValueError("regression pointer must be a dict")

    if meta_obj.report_type != "post_close" or meta_obj.product_mode != "observation":
        raise ValueError("regression metadata must be post_close observation mode")
    report_identity = report_obj.identity
    artifact_identity = artifact_obj.identity
    if report_identity.product_mode != "observation_with_research":
        raise ValueError("professional report product_mode is invalid")

    for label, actual, expected in (
        ("metadata source_market_date", meta_obj.source_market_date.isoformat(), artifact_identity.source_market_date),
        ("metadata applicable_trading_date", meta_obj.applicable_trading_date.isoformat(), artifact_identity.applicable_trading_date),
        ("metadata source_manifest", meta_obj.source_manifest, artifact_identity.source_manifest),
        ("metadata source_manifest_sha256", meta_obj.source_manifest_sha256, artifact_identity.source_manifest_sha256),
        ("report source_market_date", report_identity.source_market_date.isoformat(), artifact_identity.source_market_date),
        ("report applicable_trading_date", report_identity.applicable_trading_date.isoformat(), artifact_identity.applicable_trading_date),
        ("report source_manifest", report_identity.source_manifest, artifact_identity.source_manifest),
        ("report source_manifest_sha256", report_identity.source_manifest_sha256, artifact_identity.source_manifest_sha256),
    ):
        if actual != expected:
            raise ValueError(f"{label} mismatch")

    object_sha256 = pointer.get("sha256")
    if not isinstance(object_sha256, str) or re.fullmatch(r"[0-9a-f]{64}", object_sha256) is None:
        raise ValueError("regression pointer sha256 is invalid")
    expected_pointer = {
        "object": f"objects/regression/{object_sha256}.json",
        "sha256": object_sha256,
        "content_sha256": artifact_identity.content_sha256,
        "schema_version": 1,
        "generator_version": artifact_identity.generator_version,
        "code_commit_sha": artifact_identity.code_commit_sha,
    }
    if pointer != expected_pointer:
        raise ValueError("regression pointer does not bind exact artifact identity")
    if meta_obj.regression_research != expected_pointer:
        raise ValueError("metadata regression_research does not match pointer")

    input_sha = artifact_identity.input_dataset_sha256
    if artifact_identity.input_dataset_object != f"objects/regression-input/{input_sha}.json":
        raise ValueError("input dataset object path does not bind object SHA")

    reference = report_obj.quantitative_research.data.get("regression_reference")
    expected_status = (
        "available_with_limited_sample_warning"
        if artifact_obj.regression_spec.sample_count < 60
        else "available"
    )
    expected_reference = {
        "object_sha256": object_sha256,
        "content_sha256": artifact_identity.content_sha256,
        "summary_status": expected_status,
    }
    if report_obj.quantitative_research.status != "available" or reference != expected_reference:
        raise ValueError("professional quantitative_research regression_reference mismatch")
