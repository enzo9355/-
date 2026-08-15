"""Fail-closed contracts for the explicit TW latest-session catch-up path."""

import argparse
import datetime
import json
from dataclasses import dataclass
from pathlib import Path

from .calendar import CalendarError, TradingCalendarSet


class CatchUpContractError(ValueError):
    """The explicit catch-up request is outside the safe contract."""


@dataclass(frozen=True)
class LivePointerDate:
    name: str
    effective_date: datetime.date


def _require_date(value, label):
    if not isinstance(value, datetime.date) or isinstance(value, datetime.datetime):
        raise CatchUpContractError(f"{label} must be a date")
    return value


def latest_completed_session(calendars, *, local_today):
    """Return the latest calendar session strictly before local_today."""

    local_today = _require_date(local_today, "local_today")
    candidate = local_today - datetime.timedelta(days=1)
    while candidate >= local_today - datetime.timedelta(days=366):
        try:
            if calendars.is_session(candidate):
                return candidate
        except CalendarError as exc:
            raise CatchUpContractError(
                "calendar coverage is insufficient before local_today"
            ) from exc
        candidate -= datetime.timedelta(days=1)
    raise CatchUpContractError("no completed TW session exists before local_today")


def validate_target_session(calendars, *, target_date, local_today):
    """Validate the only date that the catch-up entry point may publish."""

    target_date = _require_date(target_date, "target_date")
    local_today = _require_date(local_today, "local_today")
    if target_date >= local_today:
        raise CatchUpContractError(
            "TargetDate must be strictly before local today"
        )
    try:
        is_session = calendars.is_session(target_date)
    except CalendarError as exc:
        raise CatchUpContractError("TargetDate is outside the TW calendar") from exc
    if not is_session:
        raise CatchUpContractError("TargetDate is not a TW trading session")
    latest = latest_completed_session(calendars, local_today=local_today)
    if target_date != latest:
        raise CatchUpContractError(
            "TargetDate is not the latest completed TW session before local today"
        )
    later = target_date + datetime.timedelta(days=1)
    while later < local_today:
        try:
            if calendars.is_session(later):
                raise CatchUpContractError(
                    "a later completed TW session exists before local today"
                )
        except CalendarError as exc:
            raise CatchUpContractError(
                "calendar coverage is insufficient after TargetDate"
            ) from exc
        later += datetime.timedelta(days=1)
    return {
        "target_date": target_date.isoformat(),
        "local_today": local_today.isoformat(),
        "latest_completed_session": latest.isoformat(),
    }


def classify_live_pointer_dates(pointers, *, target_date, local_today):
    """Classify a coherent live pointer set as publishable or idempotent."""

    target_date = _require_date(target_date, "target_date")
    local_today = _require_date(local_today, "local_today")
    values = tuple(pointers)
    if not values:
        raise CatchUpContractError("live pointer set is empty")
    if any(not isinstance(item, LivePointerDate) for item in values):
        raise CatchUpContractError("live pointer identity is invalid")
    if len({item.name for item in values}) != len(values):
        raise CatchUpContractError("live pointer names are duplicated")
    if any(item.effective_date > local_today for item in values):
        raise CatchUpContractError("a live pointer is from the future")
    dates = {item.effective_date for item in values}
    if len(dates) != 1:
        raise CatchUpContractError("live pointers are not date-coherent")
    live_date = next(iter(dates))
    if live_date > target_date:
        raise CatchUpContractError("a live pointer is newer than TargetDate")
    if live_date == target_date:
        return {"mode": "idempotent", "live_date": live_date.isoformat()}
    return {"mode": "publish", "live_date": live_date.isoformat()}


def _load_calendars(paths):
    try:
        documents = [
            json.loads(Path(path).read_text(encoding="utf-8")) for path in paths
        ]
        return TradingCalendarSet.from_documents(documents)
    except (OSError, UnicodeError, ValueError, CalendarError) as exc:
        raise CatchUpContractError("TW calendar artifact is invalid") from exc


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Validate the explicit TW latest-completed-session catch-up contract"
    )
    parser.add_argument("--target-date", required=True, type=datetime.date.fromisoformat)
    parser.add_argument(
        "--local-today", required=True, type=datetime.date.fromisoformat
    )
    parser.add_argument("--calendar-artifact", action="append", required=True)
    parser.add_argument(
        "--live-pointer",
        action="append",
        default=[],
        metavar="NAME=YYYY-MM-DD",
    )
    args = parser.parse_args(argv)
    calendars = _load_calendars(args.calendar_artifact)
    session = validate_target_session(
        calendars,
        target_date=args.target_date,
        local_today=args.local_today,
    )
    pointers = []
    for raw in args.live_pointer:
        name, separator, date_text = raw.partition("=")
        if not separator or not name:
            raise CatchUpContractError("live pointer argument is invalid")
        try:
            effective_date = datetime.date.fromisoformat(date_text)
        except ValueError as exc:
            raise CatchUpContractError("live pointer date is invalid") from exc
        pointers.append(LivePointerDate(name, effective_date))
    result = dict(session)
    if pointers:
        result["live"] = classify_live_pointer_dates(
            pointers,
            target_date=args.target_date,
            local_today=args.local_today,
        )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
