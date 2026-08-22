import datetime
import hashlib
import json
import os
import re
from pathlib import Path
import tempfile
from typing import Any

from . import REPORT_GENERATOR_VERSION, REPORT_SCHEMA_VERSION, git_commit_sha
from .config import MAX_CANONICAL_REPORT_BYTES, ReportConfig
from .exceptions import ReportPublishError
from .public_report import build_public_report
from .professional_binding import (
    validate_professional_report_binding,
    validate_regression_research_binding,
)
from .professional_schema import ProfessionalPostCloseReport, compute_content_sha256
from .publish_lock import report_v2_publish_lock
from .regression_schema import (
    MAX_REGRESSION_ARTIFACT_BYTES,
    RegressionResearchArtifact,
    serialize_regression_artifact,
)
from .schemas import (
    DailyIndustryReport,
    LoadedReportSource,
    ReportGenerationResult,
    ReportMetadataV2,
)
from .web import validate_report_index


def _json_bytes(document: dict) -> bytes:
    return json.dumps(
        document,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    ).encode("utf-8")


def _write_temporary(path: Path, content: bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="wb",
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
        delete=False,
    ) as stream:
        temporary = Path(stream.name)
        stream.write(content)
        stream.flush()
        os.fsync(stream.fileno())
    return temporary


def _write_atomic(path: Path, content: bytes) -> None:
    temporary = _write_temporary(path, content)
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _write_immutable(path: Path, content: bytes, conflict_message: str) -> bool:
    """Publish complete bytes without replacing another writer's object."""
    temporary = _write_temporary(path, content)
    try:
        try:
            os.link(temporary, path)
        except FileExistsError:
            if path.read_bytes() != content:
                raise ReportPublishError(conflict_message)
            return False
        return True
    finally:
        temporary.unlink(missing_ok=True)


_RESTORE_WRITE_ATOMIC = _write_atomic


def _local_mirror_path(archive: Path, market: str, report_date: datetime.date) -> Path:
    return Path(archive) / f"absorb-{market.lower()}-industry-daily-{report_date.isoformat()}.pdf"


def _mirror_matches(pdf_path: Path, sidecar_path: Path, pdf_sha256: str) -> bool:
    try:
        if hashlib.sha256(pdf_path.read_bytes()).hexdigest() != pdf_sha256:
            return False
        return json.loads(sidecar_path.read_text(encoding="utf-8")).get("pdf_sha256") == pdf_sha256
    except (OSError, ValueError, TypeError):
        return False


def _restore_atomic(path: Path, previous: bytes | None) -> None:
    if previous is None:
        path.unlink(missing_ok=True)
    else:
        _write_atomic(path, previous)


def _publish_report_v2_impl(
    root: Path,
    metadata: dict,
    *,
    pdf_path: Path | None = None,
    page_count: int | None = None,
    config: ReportConfig | None = None,
    professional_report: ProfessionalPostCloseReport | None = None,
    regression_artifact: dict | Any | None = None,
    _transaction_created: list[Path] | None = None,
) -> Path:
    """發布 v2 post-close、pre-market 或 weekly report；latest 永遠最後寫。"""
    settings = config or ReportConfig(root=Path(root))
    
    # 1. 驗證 Observation Metadata 結構
    try:
        schema = ReportMetadataV2.from_document(metadata)
    except ValueError as exc:
        raise ValueError(str(exc)) from exc
    if schema.report_type == "pre_market" and pdf_path is not None:
        raise ValueError("pre_market report must not include PDF")
    if pdf_path is None and page_count is not None:
        raise ValueError("page_count requires PDF")
    if schema.report_type != "post_close" and professional_report is not None:
        raise ValueError("only post_close can include professional_report")
    if schema.report_type != "post_close" and regression_artifact is not None:
        raise ValueError("only post_close can include regression_artifact")
    if regression_artifact is not None and professional_report is None:
        raise ValueError("regression_artifact requires professional_report binding")

    # 2. 處理 PDF
    pdf_bytes = None
    pdf_sha = None
    if pdf_path is not None:
        if type(page_count) is not int or page_count < 1:
            raise ValueError("PDF page_count is invalid")
        try:
            pdf_bytes = Path(pdf_path).read_bytes()
        except OSError as exc:
            raise ReportPublishError("report v2 PDF is unavailable") from exc
        if not 0 < len(pdf_bytes) <= settings.max_pdf_bytes:
            raise ReportPublishError("report v2 PDF size is invalid")
        pdf_sha = hashlib.sha256(pdf_bytes).hexdigest()

    publish = Path(root) / "publish" / "reports" / "v2"
    document = schema.to_document()
    document["content_sha256"] = hashlib.sha256(
        _json_bytes(document["content"])
    ).hexdigest()
    
    reg_obj: RegressionResearchArtifact | None = None

    # 2b. Process Regression Artifact (if provided)
    if regression_artifact is not None:
        try:
            if isinstance(regression_artifact, RegressionResearchArtifact):
                reg_obj = regression_artifact
                reg_doc = reg_obj.to_document()
            elif isinstance(regression_artifact, dict):
                reg_doc = regression_artifact
                reg_obj = RegressionResearchArtifact.from_document(reg_doc)
            else:
                raise TypeError("regression_artifact must be RegressionResearchArtifact or dict")

            reg_bytes = serialize_regression_artifact(reg_doc)
            if not 0 < len(reg_bytes) <= MAX_REGRESSION_ARTIFACT_BYTES:
                raise ReportPublishError("regression artifact object size invalid")
            reg_sha = hashlib.sha256(reg_bytes).hexdigest()
        except ReportPublishError:
            raise
        except Exception as exc:
            raise ReportPublishError("failed to process regression artifact") from exc

        reg_relative = f"objects/regression/{reg_sha}.json"
        reg_path = publish / reg_relative

        if reg_path.exists():
            if reg_path.read_bytes() != reg_bytes:
                raise ReportPublishError("immutable regression object conflict")
        else:
            try:
                created_by_this_transaction = _write_immutable(
                    reg_path,
                    reg_bytes,
                    "immutable regression object conflict",
                )
                if created_by_this_transaction:
                    if _transaction_created is not None:
                        _transaction_created.append(reg_path)
                readback_bytes = reg_path.read_bytes()
                if not 0 < len(readback_bytes) <= MAX_REGRESSION_ARTIFACT_BYTES:
                    raise ReportPublishError("regression read-back size invalid")
                if hashlib.sha256(readback_bytes).hexdigest() != reg_sha:
                    raise ReportPublishError("regression object read-back verification failed")
                readback_doc = json.loads(readback_bytes.decode("utf-8"))
                readback_obj = RegressionResearchArtifact.from_document(readback_doc)
                if readback_obj.identity.content_sha256 != reg_obj.identity.content_sha256:
                    raise ReportPublishError("regression content_sha256 read-back mismatch")
            except ReportPublishError:
                raise
            except Exception as exc:
                raise ReportPublishError("failed to write/verify regression object") from exc

        document["regression_research"] = {
            "object": reg_relative,
            "sha256": reg_sha,
            "content_sha256": reg_doc["identity"]["content_sha256"],
            "schema_version": 1,
            "generator_version": reg_doc["identity"]["generator_version"],
            "code_commit_sha": reg_doc["identity"]["code_commit_sha"],
        }
        try:
            schema = ReportMetadataV2.from_document(document)
            document = schema.to_document()
        except ValueError as exc:
            raise ReportPublishError("invalid metadata after injecting regression_research") from exc

    # 3. Process Canonical Report (ProfessionalPostCloseReport)
    if professional_report is not None:
        try:
            if not isinstance(professional_report, ProfessionalPostCloseReport):
                raise TypeError("professional_report must be ProfessionalPostCloseReport")

            canonical_doc = professional_report.to_document()
            if reg_obj is not None:
                sample_count = reg_obj.regression_spec.sample_count
                canonical_doc["quantitative_research"] = {
                    "status": "available",
                    "data_as_of": reg_obj.identity.source_market_date,
                    "data": {
                        "regression_reference": {
                            "object_sha256": document["regression_research"]["sha256"],
                            "content_sha256": reg_obj.identity.content_sha256,
                            "summary_status": (
                                "available_with_limited_sample_warning"
                                if sample_count < 60
                                else "available"
                            ),
                        }
                    },
                }
                canonical_doc["identity"]["content_sha256"] = compute_content_sha256(
                    canonical_doc
                )
                canonical_obj = ProfessionalPostCloseReport.from_document(canonical_doc)
            else:
                canonical_obj = professional_report
            validate_professional_report_binding(metadata=schema, report=canonical_obj)
            canonical_doc = canonical_obj.to_document()
            if reg_obj is not None:
                validate_regression_research_binding(
                    metadata=schema,
                    professional_report=canonical_obj,
                    pointer=document["regression_research"],
                    regression_artifact=reg_obj,
                )
            recalc_sha = compute_content_sha256(canonical_doc)
            if canonical_doc["identity"]["content_sha256"] != recalc_sha:
                raise ValueError("canonical report content_sha256 mismatch")
            canonical_bytes = _json_bytes(canonical_doc)
            if not 0 < len(canonical_bytes) <= MAX_CANONICAL_REPORT_BYTES:
                raise ReportPublishError("canonical report object size invalid")
            canonical_sha = hashlib.sha256(canonical_bytes).hexdigest()
        except ReportPublishError:
            raise
        except Exception as exc:
            raise ReportPublishError("failed to process canonical report") from exc

        canonical_relative = f"objects/canonical/{canonical_sha}.json"
        canonical_path = publish / canonical_relative

        # 4. Write Canonical Object and verify
        if canonical_path.exists():
            if canonical_path.read_bytes() != canonical_bytes:
                raise ReportPublishError("immutable canonical object conflict")
        else:
            try:
                created_by_this_transaction = _write_immutable(
                    canonical_path,
                    canonical_bytes,
                    "immutable canonical object conflict",
                )
                if created_by_this_transaction:
                    if _transaction_created is not None:
                        _transaction_created.append(canonical_path)
                readback_bytes = canonical_path.read_bytes()
                if not 0 < len(readback_bytes) <= MAX_CANONICAL_REPORT_BYTES:
                    raise ReportPublishError("canonical read-back size invalid")
                if hashlib.sha256(readback_bytes).hexdigest() != canonical_sha:
                    raise ReportPublishError("canonical object read-back verification failed")
                readback_doc = json.loads(readback_bytes.decode("utf-8"))
                readback_obj = ProfessionalPostCloseReport.from_document(readback_doc)
                validate_professional_report_binding(metadata=schema, report=readback_obj)
            except ReportPublishError:
                raise
            except Exception as exc:
                raise ReportPublishError("failed to write/verify canonical object") from exc

        # 5. Update metadata pointer
        document["professional_report"] = {
            "object": canonical_relative,
            "sha256": canonical_sha,
            "content_sha256": canonical_doc["identity"]["content_sha256"],
            "schema_version": 1,
            "generator_version": canonical_doc["identity"]["generator_version"],
            "code_commit_sha": canonical_doc["identity"]["code_commit_sha"],
        }
        # re-validate after injection
        try:
            schema = ReportMetadataV2.from_document(document)
            document = schema.to_document()
        except ValueError as exc:
            raise ReportPublishError("invalid metadata after injecting professional_report") from exc

    document["content_sha256"] = hashlib.sha256(
        _json_bytes(document["content"])
    ).hexdigest()

    if pdf_bytes is not None:
        document.update(
            pdf_path=f"objects/{pdf_sha}.pdf",
            pdf_sha256=pdf_sha,
            pdf_size=len(pdf_bytes),
            page_count=page_count,
        )
    metadata_bytes = _json_bytes(document)
    metadata_sha = hashlib.sha256(metadata_bytes).hexdigest()
    metadata_relative = f"metadata/{metadata_sha}.json"
    market = document.get("market", "TW")
    index_path = publish / f"index-{market}.json"
    previous_index = index_path.read_bytes() if index_path.exists() else None
    if previous_index is not None:
        try:
            reports = validate_report_index(previous_index, settings)
        except Exception as exc:
            raise ReportPublishError("existing v2 report index is invalid") from exc
    else:
        reports = []

    entry = {
        "report_type": document["report_type"],
        "source_market_date": document["source_market_date"],
        "applicable_trading_date": document["applicable_trading_date"],
        "published_at": document["published_at"],
        "data_as_of": document["data_as_of"],
        "model_versions": document["model_versions"],
        "title": document["title"],
        "summary": document["summary"],
        "content_sha256": document["content_sha256"],
        "metadata": metadata_relative,
        "metadata_sha256": metadata_sha,
    }
    if document.get("product_mode") is not None:
        entry["product_mode"] = document["product_mode"]
    if document["report_type"] == "weekly_model":
        week_id = document["content"].get("week_id")
        if not isinstance(week_id, str) or re.fullmatch(r"[0-9]{4}-W[0-9]{2}", week_id) is None:
            raise ReportPublishError("weekly report week_id is invalid")
        entry["week_id"] = week_id
    if pdf_bytes is not None:
        entry.update(
            pdf_path=document["pdf_path"],
            pdf_sha256=pdf_sha,
            pdf_size=len(pdf_bytes),
            page_count=page_count,
        )
    logical_key = (
        entry["report_type"],
        entry["source_market_date"],
        entry["applicable_trading_date"],
    )
    existing = [
        item
        for item in reports
        if (
            item["report_type"],
            item["source_market_date"],
            item["applicable_trading_date"],
        )
        == logical_key
    ]
    if existing and existing != [entry]:
        if document.get("product_mode") == "observation":
            reports = [
                item
                for item in reports
                if (
                    item["report_type"],
                    item["source_market_date"],
                    item["applicable_trading_date"],
                )
                != logical_key
            ]
            reports.append(entry)
        else:
            raise ReportPublishError("conflicting report v2 content")

    if pdf_bytes is not None:
        object_path = publish / document["pdf_path"]
        if object_path.exists():
            if object_path.read_bytes() != pdf_bytes:
                raise ReportPublishError("immutable report v2 PDF conflict")
        else:
            created_by_this_transaction = _write_immutable(
                object_path,
                pdf_bytes,
                "immutable report v2 PDF conflict",
            )
            if created_by_this_transaction and _transaction_created is not None:
                _transaction_created.append(object_path)
    metadata_path = publish / metadata_relative
    if metadata_path.exists():
        if metadata_path.read_bytes() != metadata_bytes:
            raise ReportPublishError("immutable report v2 metadata conflict")
    else:
        try:
            created_by_this_transaction = _write_immutable(
                metadata_path,
                metadata_bytes,
                "immutable report v2 metadata conflict",
            )
            if created_by_this_transaction:
                if _transaction_created is not None:
                    _transaction_created.append(metadata_path)
            readback_meta = metadata_path.read_bytes()
            if hashlib.sha256(readback_meta).hexdigest() != metadata_sha:
                raise ReportPublishError("metadata read-back SHA256 mismatch")
        except ReportPublishError:
            raise
        except Exception as exc:
            raise ReportPublishError("failed to write/verify metadata") from exc

    if not existing:
        reports.append(entry)
    reports.sort(key=lambda item: item["published_at"], reverse=True)
    index = {
        "schema_version": 2,
        "kind": "absorb-report-index",
        "market": market,
        "updated_at": reports[0]["published_at"],
        "reports": reports[: settings.index_history_days * 3],
    }
    index_bytes = _json_bytes(index)
    if len(index_bytes) > settings.max_index_bytes:
        raise ReportPublishError("report v2 index exceeds size limit")
    latest = {
        "schema_version": 2,
        "kind": "absorb-report",
        "market": market,
        "report_type": document["report_type"],
        "source_market_date": document["source_market_date"],
        "applicable_trading_date": document["applicable_trading_date"],
        "published_at": document["published_at"],
        "metadata": metadata_relative,
        "metadata_sha256": metadata_sha,
    }
    if document.get("product_mode") is not None:
        latest["product_mode"] = document["product_mode"]
    latest_path = publish / f"latest-{market}-{document['report_type']}.json"
    _write_atomic(index_path, index_bytes)
    _write_atomic(latest_path, _json_bytes(latest))
    return latest_path


def publish_report_v2(
    root: Path,
    metadata: dict,
    *,
    pdf_path: Path | None = None,
    page_count: int | None = None,
    config: ReportConfig | None = None,
    professional_report: ProfessionalPostCloseReport | None = None,
    regression_artifact: dict | Any | None = None,
) -> Path:
    """Run report-v2 publication inside one rollback boundary."""
    with report_v2_publish_lock(Path(root)):
        schema = ReportMetadataV2.from_document(metadata)
        publish = Path(root) / "publish" / "reports" / "v2"
        index_path = publish / f"index-{schema.market}.json"
        latest_path = publish / f"latest-{schema.market}-{schema.report_type}.json"
        previous_index = index_path.read_bytes() if index_path.exists() else None
        previous_latest = latest_path.read_bytes() if latest_path.exists() else None
        newly_created_paths: list[Path] = []

        try:
            return _publish_report_v2_impl(
                root,
                metadata,
                pdf_path=pdf_path,
                page_count=page_count,
                config=config,
                professional_report=professional_report,
                regression_artifact=regression_artifact,
                _transaction_created=newly_created_paths,
            )
        except Exception as original:
            rollback_errors = []
            for path in reversed(newly_created_paths):
                try:
                    path.unlink(missing_ok=True)
                except OSError as exc:
                    rollback_errors.append(exc)
            for path, previous in (
                (index_path, previous_index),
                (latest_path, previous_latest),
            ):
                try:
                    if previous is None:
                        path.unlink(missing_ok=True)
                    else:
                        _RESTORE_WRITE_ATOMIC(path, previous)
                except OSError as exc:
                    rollback_errors.append(exc)
            if rollback_errors:
                raise ReportPublishError("report v2 rollback failed") from original
            raise


def _write_local_mirror(
    archive: Path,
    market: str,
    report_date: datetime.date,
    pdf_bytes: bytes,
    metadata_bytes: bytes,
    pdf_sha256: str,
) -> Path:
    """將已發布的 PDF 與 metadata 寫入可讀鏡像，失敗時還原既有副本。"""
    pdf_path = _local_mirror_path(archive, market, report_date)
    sidecar_path = pdf_path.with_suffix(".json")
    if _mirror_matches(pdf_path, sidecar_path, pdf_sha256):
        return pdf_path
    previous_pdf = None
    previous_sidecar = None
    try:
        previous_pdf = pdf_path.read_bytes() if pdf_path.exists() else None
        previous_sidecar = sidecar_path.read_bytes() if sidecar_path.exists() else None
        _write_atomic(pdf_path, pdf_bytes)
        _write_atomic(sidecar_path, metadata_bytes)
    except OSError as exc:
        try:
            _restore_atomic(pdf_path, previous_pdf)
            _restore_atomic(sidecar_path, previous_sidecar)
        except OSError:
            pass
        raise ReportPublishError("local report mirror update failed") from exc
    return pdf_path


def is_source_already_published(root: Path, source: LoadedReportSource) -> bool:
    """判斷同交易日、同來源 manifest 是否已正式發布。"""
    publish = Path(root) / "publish" / "reports" / "v1"
    latest_path = publish / "latest-TW.json"
    try:
        latest = json.loads(latest_path.read_text(encoding="utf-8"))
        metadata_path = publish / str(latest["metadata"])
        metadata_bytes = metadata_path.read_bytes()
        if hashlib.sha256(metadata_bytes).hexdigest() != latest["metadata_sha256"]:
            return False
        metadata = json.loads(metadata_bytes)
        return (
            metadata.get("report_date") == source.manifest.market_as_of.isoformat()
            and metadata.get("source_manifest_sha256") == source.manifest.manifest_sha256
        )
    except (KeyError, OSError, TypeError, ValueError):
        return False


def publish_report(
    root: Path,
    report: DailyIndustryReport,
    result: ReportGenerationResult,
    config: ReportConfig | None = None,
    archive_dir: Path | None = None,
) -> Path:
    """以 content-addressed PDF/metadata 更新 index，最後替換 latest。"""
    settings = config or ReportConfig(root=Path(root))
    if not result.success or result.output_path is None or result.sha256 is None:
        raise ValueError("only a successful report can be published")
    if any(stock.sample_data for stock in report.source.stocks):
        raise ValueError("SAMPLE / TEST DATA 不得正式發布")
    try:
        pdf_bytes = result.output_path.read_bytes()
    except OSError as exc:
        raise ReportPublishError("generated PDF is unavailable") from exc
    if (
        len(pdf_bytes) != result.file_size
        or not 0 < len(pdf_bytes) <= settings.max_pdf_bytes
        or hashlib.sha256(pdf_bytes).hexdigest() != result.sha256
    ):
        raise ReportPublishError("generated PDF size or hash mismatch")

    publish = Path(root) / "publish" / "reports" / "v1"
    pdf_relative = f"objects/{result.sha256}.pdf"
    pdf_path = publish / pdf_relative
    if pdf_path.exists():
        if pdf_path.read_bytes() != pdf_bytes:
            raise ReportPublishError("immutable PDF object conflict")
    else:
        _write_immutable(pdf_path, pdf_bytes, "immutable PDF object conflict")

    manifest = report.source.manifest
    public_report = build_public_report(report)
    metadata = {
        "schema_version": 1,
        "report_schema_version": REPORT_SCHEMA_VERSION,
        "report_generator_version": REPORT_GENERATOR_VERSION,
        "git_commit_sha": git_commit_sha(),
        "sample_data": False,
        "kind": "daily-industry-report",
        "market": "TW",
        "report_date": report.report_date.isoformat(),
        "title": "ABSORB 台股產業量化分析日報",
        "generated_at": result.generated_at.isoformat().replace("+00:00", "Z"),
        "data_as_of": manifest.market_as_of.isoformat(),
        "source_manifest": f"quant/v1/{manifest.manifest_path}",
        "source_manifest_sha256": manifest.manifest_sha256,
        "model_versions": report.model_versions,
        "universe_count": manifest.universe_count,
        "symbol_count": manifest.symbol_count,
        "failure_count": manifest.failure_count,
        "coverage": manifest.coverage,
        "pdf_path": pdf_relative,
        "pdf_sha256": result.sha256,
        "pdf_size": result.file_size,
        "page_count": result.page_count,
        "summary": report.summary,
        "warnings": result.warnings,
        "public_report": public_report,
    }
    metadata_bytes = _json_bytes(metadata)
    metadata_sha = hashlib.sha256(metadata_bytes).hexdigest()
    metadata_relative = f"metadata/{metadata_sha}.json"
    metadata_path = publish / metadata_relative
    if metadata_path.exists():
        if metadata_path.read_bytes() != metadata_bytes:
            raise ReportPublishError("immutable metadata conflict")
    else:
        _write_immutable(metadata_path, metadata_bytes, "immutable metadata conflict")

    entry = {
        "report_date": metadata["report_date"],
        "data_as_of": metadata["data_as_of"],
        "generated_at": metadata["generated_at"],
        "model_versions": metadata["model_versions"],
        "coverage": metadata["coverage"],
        "pdf_path": pdf_relative,
        "pdf_sha256": result.sha256,
        "pdf_size": result.file_size,
        "page_count": result.page_count,
        "market_action": public_report["market_recommendation"]["action"],
        "headline": public_report["market_recommendation"]["headline"],
        "key_industries": [
            item["name"] for item in public_report["industries"][:3]
        ],
        "metadata": metadata_relative,
        "metadata_sha256": metadata_sha,
    }
    index_path = publish / "index-TW.json"
    if index_path.exists():
        try:
            reports = validate_report_index(index_path.read_bytes(), settings)
        except Exception as exc:
            raise ReportPublishError("existing report index is invalid") from exc
    else:
        reports = []
    reports = [item for item in reports if item.get("report_date") != entry["report_date"]]
    reports.append(entry)
    reports.sort(key=lambda item: item["report_date"], reverse=True)
    index = {
        "schema_version": 1,
        "kind": "daily-industry-report-index",
        "market": "TW",
        "updated_at": metadata["generated_at"],
        "reports": reports[: settings.index_history_days],
    }
    index_bytes = _json_bytes(index)
    if len(index_bytes) > settings.max_index_bytes:
        raise ReportPublishError("report index exceeds size limit")
    _write_atomic(index_path, index_bytes)

    latest = {
        "schema_version": 1,
        "kind": "daily-industry-report",
        "market": "TW",
        "report_date": metadata["report_date"],
        "generated_at": metadata["generated_at"],
        "metadata": metadata_relative,
        "metadata_sha256": metadata_sha,
    }
    latest_path = publish / "latest-TW.json"
    _write_atomic(latest_path, _json_bytes(latest))
    _write_local_mirror(
        archive_dir or (Path(root) / "reports" / manifest.market),
        manifest.market,
        report.report_date,
        pdf_bytes,
        metadata_bytes,
        result.sha256,
    )
    return latest_path
