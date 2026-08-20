"""Canonical schema for ABSORB institutional post-close research reports.

This module contains no rendering or storage logic. It validates the single
source of truth consumed by HTML, PDF, LINE, and Gemini adapters.
"""

from __future__ import annotations

import copy
import datetime as dt
import hashlib
import json
import math
import re
from dataclasses import dataclass
from typing import Any, Mapping

PROFESSIONAL_REPORT_SCHEMA_VERSION = 1
PROFESSIONAL_REPORT_KIND = "absorb-professional-post-close-report"
SECTION_NAMES = (
    "market",
    "capital_flows",
    "industries",
    "securities",
    "quantitative_research",
    "validation",
    "next_session",
    "governance",
    "ai_reference",
)
CRITICAL_AVAILABLE_SECTIONS = frozenset({"market", "governance"})
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_COMMIT_SHA_RE = re.compile(r"[0-9a-f]{7,64}")
_MANIFEST_RE = re.compile(
    r"quant/v1/manifests/(?:TW|US)-[0-9]{8}T[0-9]{6}Z-[0-9a-f]{12}\.json"
)
_REPORT_ID_RE = re.compile(r"(?:TW|US)-[0-9]{8}-post-close-institutional")


def _require_string(value: Any, label: str, *, maximum: int = 500) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > maximum:
        raise ValueError(f"{label} must be a non-empty string")
    return value


def _parse_date(value: Any, label: str) -> dt.date:
    try:
        return dt.date.fromisoformat(str(value))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be an ISO date") from exc


def _parse_datetime(value: Any, label: str) -> dt.datetime:
    try:
        parsed = dt.datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be an ISO datetime") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"{label} must be timezone-aware")
    return parsed


def _datetime_document(value: dt.datetime) -> str:
    return value.astimezone(dt.timezone.utc).isoformat().replace("+00:00", "Z")


def _ensure_finite_json(value: Any, path: str = "$") -> None:
    """Reject non-JSON values and non-finite floats without coercing None."""

    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"finite JSON required at {path}")
        return
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _ensure_finite_json(item, f"{path}[{index}]")
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError(f"JSON object keys must be strings at {path}")
            _ensure_finite_json(item, f"{path}.{key}")
        return
    raise ValueError(f"finite JSON required at {path}")


def compute_content_sha256(
    document: Mapping[str, Any], *, validate_finite: bool = True
) -> str:
    """Hash canonical content while excluding the self-referential hash."""

    if not isinstance(document, Mapping):
        raise ValueError("report document must be an object")
    canonical = copy.deepcopy(dict(document))
    identity = canonical.get("identity")
    if isinstance(identity, dict):
        identity["content_sha256"] = ""
    if validate_finite:
        _ensure_finite_json(canonical)
    payload = json.dumps(
        canonical,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=not validate_finite,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True)
class ProfessionalReportIdentity:
    schema_version: int
    report_type: str
    product_tier: str
    product_mode: str
    market: str
    source_market_date: dt.date
    applicable_trading_date: dt.date
    published_at: dt.datetime
    generated_at: dt.datetime
    source_manifest: str
    source_manifest_sha256: str
    content_sha256: str
    report_id: str
    generator_version: str
    code_commit_sha: str
    model_version: str | None
    feature_schema_version: str
    recommendation_policy_version: str

    @classmethod
    def from_document(cls, document: Mapping[str, Any]) -> "ProfessionalReportIdentity":
        if not isinstance(document, Mapping):
            raise ValueError("identity must be an object")
        source_date = _parse_date(document.get("source_market_date"), "source_market_date")
        applicable_date = _parse_date(
            document.get("applicable_trading_date"), "applicable_trading_date"
        )
        published_at = _parse_datetime(document.get("published_at"), "published_at")
        generated_at = _parse_datetime(document.get("generated_at"), "generated_at")
        source_manifest = str(document.get("source_manifest") or "")
        source_manifest_sha256 = str(document.get("source_manifest_sha256") or "")
        content_sha256 = str(document.get("content_sha256") or "")
        code_commit_sha = str(document.get("code_commit_sha") or "")
        report_id = str(document.get("report_id") or "")
        model_version = document.get("model_version")
        if model_version is not None:
            model_version = _require_string(model_version, "model_version", maximum=100)

        if document.get("schema_version") != PROFESSIONAL_REPORT_SCHEMA_VERSION:
            raise ValueError("identity schema_version is unsupported")
        if document.get("report_type") != "post_close":
            raise ValueError("identity report_type must be post_close")
        if document.get("product_tier") != "institutional":
            raise ValueError("identity product_tier must be institutional")
        if document.get("product_mode") != "observation_with_research":
            raise ValueError("identity product_mode must be observation_with_research")
        if document.get("market") not in ("TW", "US"):
            raise ValueError("identity market must be TW or US")
        if source_date > applicable_date or generated_at > published_at:
            raise ValueError("identity date semantics are invalid")
        if _MANIFEST_RE.fullmatch(source_manifest) is None:
            raise ValueError(f"source_manifest is invalid: {source_manifest}")
        if _SHA256_RE.fullmatch(source_manifest_sha256) is None:
            raise ValueError("source_manifest_sha256 is invalid")
        if _SHA256_RE.fullmatch(content_sha256) is None:
            raise ValueError("content_sha256 is invalid")
        if _COMMIT_SHA_RE.fullmatch(code_commit_sha) is None:
            raise ValueError("code_commit_sha is invalid")
        if _REPORT_ID_RE.fullmatch(report_id) is None:
            raise ValueError("report_id is invalid")

        return cls(
            schema_version=PROFESSIONAL_REPORT_SCHEMA_VERSION,
            report_type="post_close",
            product_tier="institutional",
            product_mode="observation_with_research",
            market=str(document.get("market") or "TW"),
            source_market_date=source_date,
            applicable_trading_date=applicable_date,
            published_at=published_at,
            generated_at=generated_at,
            source_manifest=source_manifest,
            source_manifest_sha256=source_manifest_sha256,
            content_sha256=content_sha256,
            report_id=report_id,
            generator_version=_require_string(
                document.get("generator_version"), "generator_version", maximum=100
            ),
            code_commit_sha=code_commit_sha,
            model_version=model_version,
            feature_schema_version=_require_string(
                document.get("feature_schema_version"),
                "feature_schema_version",
                maximum=100,
            ),
            recommendation_policy_version=_require_string(
                document.get("recommendation_policy_version"),
                "recommendation_policy_version",
                maximum=100,
            ),
        )

    def to_document(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "report_type": self.report_type,
            "product_tier": self.product_tier,
            "product_mode": self.product_mode,
            "market": self.market,
            "source_market_date": self.source_market_date.isoformat(),
            "applicable_trading_date": self.applicable_trading_date.isoformat(),
            "published_at": _datetime_document(self.published_at),
            "generated_at": _datetime_document(self.generated_at),
            "source_manifest": self.source_manifest,
            "source_manifest_sha256": self.source_manifest_sha256,
            "content_sha256": self.content_sha256,
            "report_id": self.report_id,
            "generator_version": self.generator_version,
            "code_commit_sha": self.code_commit_sha,
            "model_version": self.model_version,
            "feature_schema_version": self.feature_schema_version,
            "recommendation_policy_version": self.recommendation_policy_version,
        }


@dataclass(frozen=True)
class ProfessionalExecutiveSummary:
    market_state: str
    one_line_conclusion: str
    supporting_evidence: tuple[str, ...]
    opposing_evidence: tuple[str, ...]
    largest_risk: str
    strongest_industries: tuple[str, ...]
    weakest_industries: tuple[str, ...]
    next_session_watch_conditions: tuple[str, ...]
    ai_reference_summary: str | None

    @staticmethod
    def _string_list(
        document: Mapping[str, Any], key: str, limit: int
    ) -> tuple[str, ...]:
        values = document.get(key)
        if not isinstance(values, list) or len(values) > limit:
            raise ValueError(f"executive_summary.{key} is invalid")
        if not all(
            isinstance(item, str) and item.strip() and len(item) <= 500
            for item in values
        ):
            raise ValueError(f"executive_summary.{key} is invalid")
        return tuple(values)

    @classmethod
    def from_document(
        cls, document: Mapping[str, Any]
    ) -> "ProfessionalExecutiveSummary":
        if not isinstance(document, Mapping):
            raise ValueError("executive_summary must be an object")
        ai_summary = document.get("ai_reference_summary")
        if ai_summary is not None:
            ai_summary = _require_string(
                ai_summary, "executive_summary.ai_reference_summary", maximum=1000
            )
        return cls(
            market_state=_require_string(
                document.get("market_state"),
                "executive_summary.market_state",
                maximum=100,
            ),
            one_line_conclusion=_require_string(
                document.get("one_line_conclusion"),
                "executive_summary.one_line_conclusion",
                maximum=1000,
            ),
            supporting_evidence=cls._string_list(
                document, "supporting_evidence", 10
            ),
            opposing_evidence=cls._string_list(document, "opposing_evidence", 10),
            largest_risk=_require_string(
                document.get("largest_risk"),
                "executive_summary.largest_risk",
                maximum=500,
            ),
            strongest_industries=cls._string_list(
                document, "strongest_industries", 10
            ),
            weakest_industries=cls._string_list(
                document, "weakest_industries", 10
            ),
            next_session_watch_conditions=cls._string_list(
                document, "next_session_watch_conditions", 20
            ),
            ai_reference_summary=ai_summary,
        )

    def to_document(self) -> dict[str, Any]:
        return {
            "market_state": self.market_state,
            "one_line_conclusion": self.one_line_conclusion,
            "supporting_evidence": list(self.supporting_evidence),
            "opposing_evidence": list(self.opposing_evidence),
            "largest_risk": self.largest_risk,
            "strongest_industries": list(self.strongest_industries),
            "weakest_industries": list(self.weakest_industries),
            "next_session_watch_conditions": list(
                self.next_session_watch_conditions
            ),
            "ai_reference_summary": self.ai_reference_summary,
        }


@dataclass(frozen=True)
class ProfessionalSection:
    status: str
    data: dict[str, Any]
    data_as_of: dt.date | None = None
    reason: str | None = None

    @classmethod
    def from_document(
        cls, document: Mapping[str, Any], name: str
    ) -> "ProfessionalSection":
        if not isinstance(document, Mapping):
            raise ValueError(f"{name} must be an object")
        status = document.get("status")
        data = document.get("data")
        if status not in {"available", "unavailable"} or not isinstance(data, dict):
            raise ValueError(f"{name} status/data are invalid")
        _ensure_finite_json(data, f"$.{name}.data")
        if status == "available":
            data_as_of = _parse_date(
                document.get("data_as_of"), f"{name}.data_as_of"
            )
            reason = None
        else:
            reason = _require_string(
                document.get("reason"), f"{name}.reason", maximum=500
            )
            data_as_of = None
        return cls(
            status=status,
            data=dict(data),
            data_as_of=data_as_of,
            reason=reason,
        )

    def to_document(self) -> dict[str, Any]:
        document: dict[str, Any] = {
            "status": self.status,
            "data": copy.deepcopy(self.data),
        }
        if self.status == "available":
            document["data_as_of"] = (
                self.data_as_of.isoformat() if self.data_as_of else None
            )
        else:
            document["reason"] = self.reason
        return document


@dataclass(frozen=True)
class ProfessionalPostCloseReport:
    identity: ProfessionalReportIdentity
    executive_summary: ProfessionalExecutiveSummary
    key_events: tuple[dict[str, Any], ...]
    market: ProfessionalSection
    capital_flows: ProfessionalSection
    industries: ProfessionalSection
    securities: ProfessionalSection
    quantitative_research: ProfessionalSection
    validation: ProfessionalSection
    next_session: ProfessionalSection
    governance: ProfessionalSection
    ai_reference: ProfessionalSection

    @classmethod
    def from_document(
        cls, document: Mapping[str, Any]
    ) -> "ProfessionalPostCloseReport":
        if not isinstance(document, Mapping):
            raise ValueError("professional report must be an object")
        if document.get("schema_version") != PROFESSIONAL_REPORT_SCHEMA_VERSION:
            raise ValueError("professional report schema_version is unsupported")
        if document.get("kind") != PROFESSIONAL_REPORT_KIND:
            raise ValueError("professional report kind is invalid")
        _ensure_finite_json(document)

        identity = ProfessionalReportIdentity.from_document(document.get("identity"))
        expected_hash = compute_content_sha256(document)
        if identity.content_sha256 != expected_hash:
            raise ValueError("content_sha256 does not match canonical content")

        executive_summary = ProfessionalExecutiveSummary.from_document(
            document.get("executive_summary")
        )
        key_events = document.get("key_events")
        if not isinstance(key_events, list) or len(key_events) > 8:
            raise ValueError("key_events must be a list with at most 8 items")
        if not all(isinstance(item, dict) for item in key_events):
            raise ValueError("key_events items must be objects")

        sections = {
            name: ProfessionalSection.from_document(document.get(name), name)
            for name in SECTION_NAMES
        }
        for name in CRITICAL_AVAILABLE_SECTIONS:
            if sections[name].status != "available":
                raise ValueError(f"critical section {name} must be available")
            if sections[name].data_as_of != identity.source_market_date:
                raise ValueError(
                    f"critical section {name} date must match source_market_date"
                )

        return cls(
            identity=identity,
            executive_summary=executive_summary,
            key_events=tuple(copy.deepcopy(key_events)),
            **sections,
        )

    def to_document(self) -> dict[str, Any]:
        document = {
            "schema_version": PROFESSIONAL_REPORT_SCHEMA_VERSION,
            "kind": PROFESSIONAL_REPORT_KIND,
            "identity": self.identity.to_document(),
            "executive_summary": self.executive_summary.to_document(),
            "key_events": copy.deepcopy(list(self.key_events)),
        }
        for name in SECTION_NAMES:
            document[name] = getattr(self, name).to_document()
        _ensure_finite_json(document)
        return document
