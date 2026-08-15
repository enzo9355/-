"""Operational CLI for private TEJ research; never a TW production dependency."""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
from pathlib import Path

from .challengers import load_dataset
from .evaluation import build_split_plan
from .tej import (
    TEJ_SCHEMA_VERSION,
    TejClient,
    _write_immutable,
    build_factor_snapshot,
    compare_official_truth,
    load_tej_raw_cache,
    load_tej_private_artifact,
    normalize_pit_records,
    validate_factor_snapshot,
    validate_tej_data_root,
    write_tej_raw_cache,
)
from .tej_challenger import build_tej_challenger_frame, run_tej_challenger


def _now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat().replace(
        "+00:00", "Z"
    )


def _json(path):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"JSON input is unavailable: {path}") from exc


def _json_with_sha(path):
    path = Path(path).resolve()
    try:
        content = path.read_bytes()
        document = json.loads(content)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"JSON input is unavailable: {path}") from exc
    return document, hashlib.sha256(content).hexdigest()


def _under_root(root, path):
    root = Path(root).resolve()
    path = Path(path).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise ValueError("TEJ input path escaped the allowlisted data root") from exc
    return path


def _canonical_sha(document):
    content = json.dumps(
        document,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(content).hexdigest()


def _private_artifact(root, category, document):
    try:
        content = json.dumps(
            document,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ValueError("TEJ private artifact is not finite JSON") from exc
    digest = hashlib.sha256(content).hexdigest()
    root = Path(root).resolve()
    path = root / "research" / "tej" / "v1" / category / f"{digest}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    if len(content) > 64 * 1024 * 1024:
        raise ValueError("TEJ private artifact exceeds the configured size limit")
    _write_immutable(path, content)
    return {"path": str(path), "sha256": digest, "size": len(content)}


def _safe_summary(result):
    if not isinstance(result, dict):
        return {"status": "schema_mismatch"}
    summary = {
        key: result[key]
        for key in (
            "provider",
            "status",
            "dataset",
            "dataset_count",
            "row_count",
            "reason",
            "safe_message",
            "retry_after_seconds",
        )
        if key in result
    }
    if "entitled_datasets" in result:
        summary["entitled_datasets"] = result["entitled_datasets"]
    return summary


def _status(args):
    result = TejClient.from_env().discover()
    print(json.dumps(_safe_summary(result), ensure_ascii=False, sort_keys=True))
    return 0 if result.get("status") in {"disabled", "authentication_valid"} else 2


def _fetch(args):
    filters = _json(args.filters) if args.filters else {}
    if not isinstance(filters, dict):
        raise ValueError("--filters must contain a JSON object")
    result = TejClient.from_env().fetch_dataset(
        args.table,
        filters=filters,
        columns=args.columns or None,
    )
    if result.get("status") != "dataset_entitled":
        print(json.dumps(_safe_summary(result), ensure_ascii=False, sort_keys=True))
        return 2
    retrieved = _now()
    metadata = {
        "provider": "TEJ",
        "dataset": args.table,
        "query": {"filters": filters, "columns": args.columns or []},
        "requested_at": retrieved,
        "retrieved_at": retrieved,
        "effective_date": args.effective_date,
        "available_at": args.available_at or retrieved,
        "entity_field": args.entity_field,
        "row_count": result["row_count"],
        "schema_version": TEJ_SCHEMA_VERSION,
        "client_version": "absorb-tej-v1",
    }
    cached = write_tej_raw_cache(args.root, result["records"], metadata)
    print(json.dumps({"status": result["status"], **cached}, ensure_ascii=False, sort_keys=True))
    return 0


def _normalize(args):
    cached = load_tej_raw_cache(args.root, args.metadata)
    mapping = _json(args.mapping)
    entity_map = _json(args.entity_map)
    rows = normalize_pit_records(
        cached["payload"],
        table=cached["metadata"]["dataset"],
        payload_sha256=cached["metadata"]["payload_sha256"],
        field_map=mapping,
        entity_map=entity_map,
    )
    artifact = _private_artifact(
        args.root,
        "normalized",
        {
            "schema_version": TEJ_SCHEMA_VERSION,
            "kind": "absorb-tej-pit-normalized",
            "source_metadata_sha256": cached["metadata_sha256"],
            "field_map_sha256": _canonical_sha(mapping),
            "entity_map_sha256": _canonical_sha(entity_map),
            "row_count": len(rows),
            "rows": rows,
        },
    )
    print(json.dumps({"status": "available", **artifact, "row_count": len(rows)}, sort_keys=True))
    return 0


def _factor(args):
    normalized_artifact = load_tej_private_artifact(
        args.root,
        args.normalized,
        category="normalized",
        expected_kind="absorb-tej-pit-normalized",
    )
    source = normalized_artifact["document"]
    rows = source["rows"]
    snapshot = build_factor_snapshot(
        rows,
        as_of=args.as_of,
        effective_date=args.effective_date,
        source_normalized_sha256=normalized_artifact["sha256"],
        field_map_sha256=source["field_map_sha256"],
        entity_map_sha256=source["entity_map_sha256"],
    )
    validate_factor_snapshot(snapshot)
    artifact = _private_artifact(args.root, "factors", snapshot)
    print(json.dumps({"status": snapshot["status"], **artifact, "feature_count": len(snapshot["feature_manifest"])}, sort_keys=True))
    return 0


def _challenger(args):
    price_manifest_path = _under_root(args.root, args.price_manifest)
    frame, manifest = load_dataset(price_manifest_path)
    factor_artifact = load_tej_private_artifact(
        args.root,
        args.factors,
        category="factors",
        expected_kind="absorb-tej-factor-snapshot",
    )
    joined = build_tej_challenger_frame(
        frame,
        factor_artifact["document"],
    )
    if joined.get("status") != "available":
        result = {
            "status": joined.get("status", "BLOCKED"),
            "reason": joined.get("reason"),
            "production_eligible": False,
        }
    else:
        eligible = joined["frame"][joined["frame"]["_tej_missing_count"] == 0]
        plan = build_split_plan(eligible["source_market_date"].unique())
        result = run_tej_challenger(
            joined,
            plan,
            bootstrap_iterations=args.bootstrap_iterations,
        )
    result["lineage"] = {
        **(result.get("lineage") or {}),
        "factor_snapshot_sha256": factor_artifact["sha256"],
        "price_manifest_sha256": Path(price_manifest_path).stem,
        "price_dataset_sha256": manifest["dataset_sha256"],
    }
    artifact = _private_artifact(args.root, "challengers", result)
    print(json.dumps({"status": result["status"], **artifact}, sort_keys=True))
    return 0


def _shadow(args):
    official_document, official_sha256 = _json_with_sha(
        _under_root(args.root, args.official)
    )
    tej_document, tej_sha256 = _json_with_sha(
        _under_root(args.root, args.tej)
    )
    official = (
        official_document.get("rows", official_document)
        if isinstance(official_document, dict)
        else official_document
    )
    tej = (
        tej_document.get("rows", tej_document)
        if isinstance(tej_document, dict)
        else tej_document
    )
    result = compare_official_truth(
        official,
        tej,
        official_identity={"source": args.official_source, "sha256": official_sha256},
        tej_identity={"source": "TEJ", "sha256": tej_sha256},
        checked_at=args.checked_at or _now(),
    )
    artifact = _private_artifact(args.root, "shadow", result)
    print(json.dumps({"status": result["status"], "mismatch_count": len(result["mismatches"]), **artifact}, sort_keys=True))
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description="ABSORB private TEJ research")
    parser.add_argument("command", choices=("status", "fetch", "normalize", "factor", "challenger", "shadow"))
    parser.add_argument("--root", type=Path, default=Path(r"D:\AbsorbData"))
    parser.add_argument("--table")
    parser.add_argument("--filters")
    parser.add_argument("--columns", nargs="*")
    parser.add_argument("--effective-date")
    parser.add_argument("--available-at")
    parser.add_argument("--entity-field", default="coid")
    parser.add_argument("--metadata", type=Path)
    parser.add_argument("--mapping", type=Path)
    parser.add_argument("--entity-map", type=Path)
    parser.add_argument("--normalized", type=Path)
    parser.add_argument("--as-of")
    parser.add_argument("--price-manifest", type=Path)
    parser.add_argument("--factors", type=Path)
    parser.add_argument("--bootstrap-iterations", type=int, default=500)
    parser.add_argument("--official", type=Path)
    parser.add_argument("--tej", type=Path)
    parser.add_argument("--official-source", default="TWSE/TPEx")
    parser.add_argument("--checked-at")
    args = parser.parse_args(argv)
    try:
        args.root = validate_tej_data_root(args.root)
    except ValueError as exc:
        parser.error(str(exc))
    if args.command == "status":
        return _status(args)
    if args.command == "fetch":
        if not args.table:
            parser.error("fetch requires --table")
        return _fetch(args)
    if args.command == "normalize":
        if not all((args.metadata, args.mapping, args.entity_map)):
            parser.error("normalize requires --metadata, --mapping and --entity-map")
        return _normalize(args)
    if args.command == "factor":
        if not args.normalized or not args.as_of:
            parser.error("factor requires --normalized and --as-of")
        return _factor(args)
    if args.command == "challenger":
        if not args.price_manifest or not args.factors:
            parser.error("challenger requires --price-manifest and --factors")
        return _challenger(args)
    if not args.official or not args.tej:
        parser.error("shadow requires --official and --tej")
    return _shadow(args)


if __name__ == "__main__":
    raise SystemExit(main())
