"""批次工作的獨立 namespace、檔案鎖與讓路協定。"""

import datetime
import json
import os
import re
import secrets
from dataclasses import dataclass
from pathlib import Path


JOB_TYPES = (
    "daily_observation",
    "daily_prediction",
    "post_close_report",
    "pre_market_update",
    "full_backtest",
    "full_backtest_us",
    "weekly_model_report",
    "upload",
)
DAILY_JOB_TYPES = frozenset(
    {
        "daily_observation",
        "daily_prediction",
        "post_close_report",
        "pre_market_update",
    }
)


class JobLockError(RuntimeError):
    """Job lock 已占用、損壞或 ownership 不一致。"""


def _job_type(value):
    if value not in JOB_TYPES:
        raise ValueError("unknown job type")
    return value


def _aware(value, label):
    if (
        not isinstance(value, datetime.datetime)
        or value.tzinfo is None
        or value.utcoffset() is None
    ):
        raise ValueError(f"{label} must be timezone-aware")
    return value


def _date(value):
    if type(value) is not datetime.date:
        raise ValueError("target_date must be a date")
    return value


def _timestamp(value):
    return value.astimezone(datetime.timezone.utc).isoformat().replace("+00:00", "Z")


def _write_json_exclusive(path, document):
    encoded = json.dumps(document, ensure_ascii=False, separators=(",", ":")).encode(
        "utf-8"
    )
    descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
    except Exception:
        Path(path).unlink(missing_ok=True)
        raise


def _write_json_atomic(path, document):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    encoded = json.dumps(document, ensure_ascii=False, separators=(",", ":")).encode(
        "utf-8"
    )
    with temporary.open("wb") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


@dataclass(frozen=True)
class JobNamespace:
    lock: Path
    checkpoint: Path
    status: Path
    output: Path


def job_namespace(root, job_type):
    root = Path(root)
    job_type = _job_type(job_type)
    return JobNamespace(
        lock=root / "checkpoints" / "locks" / f"{job_type}.lock.json",
        checkpoint=root / "checkpoints" / "jobs" / job_type / "current.json",
        status=root / "logs" / "pipeline-status" / f"current-{job_type}.json",
        output=root / "outputs" / job_type,
    )


def _read_lock(path, expected_job):
    try:
        document = json.loads(Path(path).read_text(encoding="utf-8"))
        started_at = datetime.datetime.fromisoformat(
            str(document["started_at"]).replace("Z", "+00:00")
        )
        updated_at = datetime.datetime.fromisoformat(
            str(document["updated_at"]).replace("Z", "+00:00")
        )
        datetime.date.fromisoformat(str(document["target_date"]))
    except (OSError, KeyError, TypeError, ValueError) as exc:
        raise JobLockError("job lock is invalid") from exc
    if (
        document.get("schema_version") != 1
        or document.get("job_type") != expected_job
        or type(document.get("pid")) is not int
        or document["pid"] < 1
        or re.fullmatch(r"[0-9a-f]{32}", str(document.get("token") or "")) is None
        or started_at.tzinfo is None
        or updated_at.tzinfo is None
        or updated_at < started_at
    ):
        raise JobLockError("job lock is invalid")
    return document, updated_at


class JobLock:
    def __init__(self, path, job_type, token):
        self.path = Path(path)
        self.job_type = job_type
        self.token = token

    def heartbeat(self, now=None):
        checked_at = _aware(now or datetime.datetime.now(datetime.timezone.utc), "now")
        document, _updated_at = _read_lock(self.path, self.job_type)
        if document["token"] != self.token:
            raise JobLockError("job lock ownership changed")
        document["updated_at"] = _timestamp(checked_at)
        _write_json_atomic(self.path, document)

    def release(self):
        if not self.path.exists():
            return
        document, _updated_at = _read_lock(self.path, self.job_type)
        if document["token"] != self.token:
            raise JobLockError("job lock ownership changed")
        self.path.unlink()

    def __enter__(self):
        return self

    def __exit__(self, _type, _value, _traceback):
        self.release()


def acquire_job_lock(
    root,
    job_type,
    target_date,
    *,
    now=None,
    pid=None,
    token=None,
    stale_after=datetime.timedelta(hours=6),
):
    job_type = _job_type(job_type)
    target_date = _date(target_date)
    checked_at = _aware(now or datetime.datetime.now(datetime.timezone.utc), "now")
    if not isinstance(stale_after, datetime.timedelta) or stale_after.total_seconds() <= 0:
        raise ValueError("stale_after must be positive")
    process_id = os.getpid() if pid is None else pid
    ownership_token = secrets.token_hex(16) if token is None else token
    if (
        type(process_id) is not int
        or process_id < 1
        or re.fullmatch(r"[0-9a-f]{32}", str(ownership_token)) is None
    ):
        raise ValueError("invalid lock owner")

    path = job_namespace(root, job_type).lock
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        _document, updated_at = _read_lock(path, job_type)
        if checked_at.astimezone(datetime.timezone.utc) - updated_at.astimezone(
            datetime.timezone.utc
        ) <= stale_after:
            raise JobLockError(f"{job_type} is already active")
        archive = path.with_name(
            f"{job_type}.lock.stale."
            f"{checked_at.astimezone(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json"
        )
        if archive.exists():
            raise JobLockError("stale lock archive already exists")
        os.replace(path, archive)

    document = {
        "schema_version": 1,
        "job_type": job_type,
        "target_date": target_date.isoformat(),
        "pid": process_id,
        "token": ownership_token,
        "started_at": _timestamp(checked_at),
        "updated_at": _timestamp(checked_at),
    }
    try:
        _write_json_exclusive(path, document)
    except FileExistsError as exc:
        raise JobLockError(f"{job_type} is already active") from exc
    return JobLock(path, job_type, ownership_token)


def yield_full_backtest_to_daily(root, *, boundary, save_checkpoint):
    if boundary not in {"symbol", "fold"}:
        raise ValueError("full backtest may yield only at symbol or fold boundaries")
    if not callable(save_checkpoint):
        raise TypeError("save_checkpoint must be callable")
    active = any(job_namespace(root, job).lock.exists() for job in DAILY_JOB_TYPES)
    if active:
        save_checkpoint("daily_pipeline_active")
        return True
    return False
