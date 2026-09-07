"""ABSORB 本機批次狀態 CLI。"""

import argparse
import datetime
import hashlib
import json
import os
import sys
from pathlib import Path
from urllib.parse import urlsplit

from stock_papi.batch.runtime import JOB_TYPES, job_namespace


def render_status(root):
    rows = []
    for job_type in JOB_TYPES:
        path = job_namespace(root, job_type).status
        if not path.exists():
            continue
        try:
            document = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            rows.append(f"{job_type}: INVALID")
            continue
        details = document.get("details") if isinstance(document.get("details"), dict) else {}
        date_value = details.get("report_date") or details.get("source_market_date") or document.get("target_date")
        error = document.get("error") or "-"
        rows.append(
            f"{job_type}: stage={document.get('stage', 'unknown')} "
            f"date={date_value or '-'} error={error}"
        )
    return "\n".join(rows) if rows else "尚無 pipeline status。"


class PreMarketRefusal(RuntimeError):
    """Machine-readable pre-market refusal with a safe category."""

    def __init__(self, category, safe_message):
        super().__init__(safe_message)
        self.category = str(category)

    def __str__(self):
        return f"{self.category}: {self.args[0]}"


def _load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _load_post_close_base(root, applicable_date):
    from reporting.web import validate_report_index, validate_report_metadata

    publish = Path(root) / "publish" / "reports" / "v2"
    reports = validate_report_index((publish / "index-TW.json").read_bytes())
    matches = [
        item
        for item in reports
        if item.get("report_type") == "post_close"
        and item.get("applicable_trading_date") == applicable_date.isoformat()
    ]
    if len(matches) != 1:
        raise ValueError("verified post-close base is unavailable")
    item = matches[0]
    metadata = validate_report_metadata(
        (publish / item["metadata"]).read_bytes(), item
    )
    return {"metadata": metadata, "metadata_sha256": item["metadata_sha256"]}


def _publish_v2_receipt(root, metadata):
    from reporting.publisher import publish_report_v2

    latest_path = publish_report_v2(Path(root), metadata)
    latest = _load_json(latest_path)
    metadata_path = latest_path.parent / latest["metadata"]
    content = metadata_path.read_bytes()
    digest = hashlib.sha256(content).hexdigest()
    if digest != latest["metadata_sha256"]:
        raise ValueError("published metadata hash mismatch")
    document = json.loads(content)
    return {
        "metadata": document,
        "metadata_sha256": digest,
        "content_sha256": document["content_sha256"],
        "latest_path": str(latest_path),
    }


def run_pre_market(args):
    if args.source_file:
        raise PreMarketRefusal(
            "unsupported_source_file",
            "TW pre-market only accepts verified US quant source; --source-file is not supported",
        )
    from stock_papi.batch.pre_market import PreMarketPipeline
    from reporting.source_loader import load_report_source
    from stock_papi.batch.calendar import TradingCalendarSet
    from stock_papi.integrations.market_data.us_calendar import get_us_calendar_documents

    calendars = None
    required_base_session = None
    if args.calendar_artifact:
        from stock_papi.batch.calendar import TradingCalendarSet

        documents = [_load_json(path) for path in args.calendar_artifact]
        calendars = TradingCalendarSet.from_documents(documents)
        required_base_session = calendars.latest_session_on_or_before(
            args.applicable_trading_date - datetime.timedelta(days=1)
        )

    loaders = []
    for source_file in args.source_file:
        path = Path(source_file)
        loaders.append(lambda path=path: _load_json(path))
    pipeline = PreMarketPipeline(
        Path(args.root),
        applicable_trading_date=args.applicable_trading_date,
        load_base=lambda: _load_post_close_base(
            Path(args.root), args.applicable_trading_date
        ),
        source_loaders=loaders,
        publish=lambda metadata: _publish_v2_receipt(Path(args.root), metadata),
        notify=lambda _receipt: {
            "enabled": False,
            "reason": "notification delivery is performed after uploader verification",
        },
        us_source_loader=lambda: load_report_source(Path(args.root), market="US"),
        us_calendars=TradingCalendarSet.from_documents(
            get_us_calendar_documents(
                args.applicable_trading_date.year - 1,
                args.applicable_trading_date.year + 1,
            )
        ),
    )
    if calendars is not None:
        base = pipeline.load_base()
        base_source = str(base["metadata"].get("source_market_date") or "")
        if base_source != required_base_session.isoformat():
            raise PreMarketRefusal(
                "stale_post_close_base",
                (
                    f"post-close base source_market_date {base_source} does not "
                    f"match the required completed session "
                    f"{required_base_session.isoformat()}"
                ),
            )
    result = pipeline.run(now=datetime.datetime.now(datetime.timezone.utc))
    print(json.dumps(result, ensure_ascii=False))
    return 0


def run_notification(args):
    if os.environ.get("REPORT_NOTIFICATION_ENABLED", "").lower() != "true":
        print(json.dumps({"enabled": False, "reason": "disabled"}))
        return 0
    from reporting.web import validate_report_index, validate_report_metadata
    from stock_papi.batch.line_delivery import line_sender
    from stock_papi.batch.notifications import NotificationManager

    root = Path(args.root)
    publish = root / "publish" / "reports" / "v2"
    reports = validate_report_index((publish / "index-TW.json").read_bytes())
    matches = [item for item in reports if item.get("report_type") == args.report_type]
    if not matches:
        raise ValueError("verified report is unavailable")
    item = matches[0]
    metadata = validate_report_metadata((publish / item["metadata"]).read_bytes(), item)
    base_url = os.environ.get("REPORT_PUBLIC_BASE_URL", "").rstrip("/")
    parsed = urlsplit(base_url)
    if parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.password:
        raise ValueError("REPORT_PUBLIC_BASE_URL must be an HTTPS origin")
    if args.report_type == "post_close":
        suffix = f"/reports/{metadata['source_market_date']}/post-close"
    elif args.report_type == "pre_market":
        suffix = f"/reports/{metadata['applicable_trading_date']}/pre-market"
    else:
        suffix = f"/reports/weekly/{metadata['content']['week_id']}"
    manager = NotificationManager(root, send=line_sender)
    receipts = []
    for audience in args.audience:
        receipts.append(
            manager.deliver(
                report_type=args.report_type,
                content_sha256=metadata["content_sha256"],
                audience=audience,
                public_url=base_url + suffix,
                summary=metadata["summary"],
                now=datetime.datetime.now(datetime.timezone.utc),
            )
        )
    print(json.dumps(receipts, ensure_ascii=False))
    return 0 if all(item.get("status") == "sent" for item in receipts) else 1


def run_calendar_check(args):
    from stock_papi.batch.calendar import TradingCalendarSet

    documents = [_load_json(path) for path in args.calendar_artifact]
    calendars = TradingCalendarSet.from_documents(documents)
    is_session = calendars.is_session(args.date)
    print(
        json.dumps(
            {"date": args.date.isoformat(), "is_session": is_session},
            ensure_ascii=False,
        )
    )
    return 0 if is_session else 3


def run_calendar_latest_session(args):
    from stock_papi.batch.calendar import TradingCalendarSet

    documents = [_load_json(path) for path in args.calendar_artifact]
    calendars = TradingCalendarSet.from_documents(documents)
    latest = calendars.latest_session_on_or_before(args.before)
    print(
        json.dumps(
            {
                "before": args.before.isoformat(),
                "latest_session": latest.isoformat(),
            },
            ensure_ascii=False,
        )
    )
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description="ABSORB batch operations")
    subparsers = parser.add_subparsers(dest="command", required=True)
    status = subparsers.add_parser("status", help="顯示所有 batch job 狀態")
    status.add_argument("--root", default=r"D:\AbsorbData")
    pre_market = subparsers.add_parser(
        "pre-market", help="由已驗證盤後 base 建立盤前 overlay"
    )
    pre_market.add_argument("--root", default=r"D:\AbsorbData")
    pre_market.add_argument(
        "--applicable-trading-date",
        type=datetime.date.fromisoformat,
        required=True,
    )
    pre_market.add_argument("--source-file", action="append", default=[])
    pre_market.add_argument("--calendar-artifact", type=Path, action="append", default=[])
    notify = subparsers.add_parser("notify", help="推送已驗證的公開報告連結")
    notify.add_argument("--root", default=r"D:\AbsorbData")
    notify.add_argument(
        "--report-type",
        choices=("post_close", "pre_market", "weekly_model"),
        required=True,
    )
    notify.add_argument(
        "--audience", choices=("admin", "broadcast"), action="append", required=True
    )
    calendar = subparsers.add_parser(
        "calendar-check", help="以已驗證 TWSE artifact 判斷交易日"
    )
    calendar.add_argument("--calendar-artifact", type=Path, action="append", required=True)
    calendar.add_argument("--date", type=datetime.date.fromisoformat, required=True)
    calendar_latest = subparsers.add_parser(
        "calendar-latest-session",
        help="以已驗證 TWSE artifact 計算基準日（含）前最近的交易日",
    )
    calendar_latest.add_argument(
        "--calendar-artifact", type=Path, action="append", required=True
    )
    calendar_latest.add_argument(
        "--before", type=datetime.date.fromisoformat, required=True
    )
    args = parser.parse_args(argv)
    if args.command == "status":
        print(render_status(Path(args.root)))
        return 0
    if args.command == "pre-market":
        try:
            return run_pre_market(args)
        except PreMarketRefusal as exc:
            print(f"TW pre-market refused: {exc}", file=sys.stderr)
            return 4
    if args.command == "notify":
        return run_notification(args)
    if args.command == "calendar-check":
        return run_calendar_check(args)
    if args.command == "calendar-latest-session":
        return run_calendar_latest_session(args)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
