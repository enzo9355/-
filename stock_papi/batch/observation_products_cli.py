"""Create and locally promote verified Observation product candidates."""

import argparse
import datetime
import json
from pathlib import Path


def _calendars(paths):
    from stock_papi.batch.calendar import TradingCalendarSet

    return TradingCalendarSet.from_documents(
        [
            json.loads(Path(path).read_text(encoding="utf-8"))
            for path in paths
        ]
    )


def build(args, *, today=None):
    from reporting.cli import _load_industry_map
    from reporting.config import ReportConfig
    from reporting.observation_v2 import build_post_close_observation_metadata
    from reporting.source_loader import load_report_source_manifest
    from stock_papi.batch.observation_products import (
        build_observation_dashboard,
        write_observation_candidate,
    )
    from stock_papi.config.capabilities import PredictionCapabilityState

    source = load_report_source_manifest(
        args.root,
        args.source_manifest,
        args.source_manifest_sha256,
        market="TW",
        report_date=args.source_market_date,
        config=ReportConfig(root=args.root, market="TW"),
    )
    capability = PredictionCapabilityState.from_environment()
    dashboard = build_observation_dashboard(
        source,
        _load_industry_map(args.root),
        capability,
        generated_at=datetime.datetime.fromisoformat(
            source.manifest.generated_at.replace("Z", "+00:00")
        ),
        today=today,
    )
    metadata = build_post_close_observation_metadata(
        dashboard,
        _calendars(args.calendar_artifact),
    )
    candidate = write_observation_candidate(args.root, metadata, dashboard)
    return {
        "mode": "observation-candidate",
        "candidate_path": str(candidate),
        "dashboard_path": str(candidate / "dashboard-snapshot.json"),
        "post_close_v2_path": str(candidate / "post-close-report-v2.json"),
        "observation_as_of": dashboard["observation_as_of"],
        "source_manifest": dashboard["source_manifest"],
        "source_manifest_sha256": dashboard["source_manifest_sha256"],
    }


def main(argv=None, *, today=None):
    parser = argparse.ArgumentParser(description="ABSORB Observation products")
    subparsers = parser.add_subparsers(dest="command", required=True)
    create = subparsers.add_parser("build")
    create.add_argument(
        "--root", type=Path, default=Path(r"D:\AbsorbData")
    )
    create.add_argument(
        "--source-market-date",
        type=datetime.date.fromisoformat,
        required=True,
    )
    create.add_argument(
        "--source-validation-date",
        type=datetime.date.fromisoformat,
        default=None,
        help="Override the freshness validation date for an explicit historical run.",
    )
    create.add_argument("--source-manifest", required=True)
    create.add_argument("--source-manifest-sha256", required=True)
    create.add_argument(
        "--calendar-artifact", type=Path, action="append", required=True
    )
    promote = subparsers.add_parser("promote")
    promote.add_argument(
        "--root", type=Path, default=Path(r"D:\AbsorbData")
    )
    promote.add_argument("--candidate", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "build":
        validation_date = (
            args.source_validation_date
            if args.source_validation_date is not None
            else today
        )
        result = build(args, today=validation_date)
    else:
        from stock_papi.batch.observation_products import (
            promote_observation_candidate,
        )

        result = {
            "mode": "observation-local-promotion",
            **promote_observation_candidate(args.root, args.candidate),
        }
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
