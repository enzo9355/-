"""Fail-closed cross-process lock for report-v2 publish transactions."""

from contextlib import contextmanager
import json
import logging
import os
from pathlib import Path
import secrets
import sys
import time
from typing import Iterator

from .exceptions import ReportPublishError


_LOCK_NAME = ".publish-transaction-lock"
_OWNER_FILE_NAME = "owner-token"
_DEFAULT_WAIT_SECONDS = 300.0
_DEFAULT_POLL_SECONDS = 0.05
_LOGGER = logging.getLogger(__name__)


def _emit_lock_receipt(
    event: str,
    *,
    waited_milliseconds: int,
    wait_seconds: float | None = None,
) -> None:
    document: dict[str, object] = {
        "kind": "absorb-pipeline-lock",
        "scope": "report_v2_publish",
        "event": event,
        "waited_milliseconds": max(0, int(waited_milliseconds)),
    }
    if wait_seconds is not None:
        document["max_wait_milliseconds"] = int(wait_seconds * 1000)
    sys.stdout.write(
        json.dumps(document, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
        + "\n"
    )
    sys.stdout.flush()


def _acquire_report_v2_publish_lock(
    publish_root: Path,
    *,
    wait_seconds: float,
    poll_seconds: float,
) -> tuple[Path, str, int]:
    lock_path = Path(publish_root) / "publish" / "reports" / "v2" / _LOCK_NAME
    owner_token = secrets.token_hex(32)
    started = time.monotonic()
    deadline = started + wait_seconds
    waiting_logged = False
    try:
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        while True:
            try:
                lock_path.mkdir()
                break
            except FileExistsError as exc:
                now = time.monotonic()
                if not waiting_logged:
                    _emit_lock_receipt(
                        "waiting",
                        waited_milliseconds=int((now - started) * 1000),
                        wait_seconds=wait_seconds,
                    )
                    waiting_logged = True
                remaining = deadline - now
                if remaining <= 0:
                    waited_milliseconds = int((now - started) * 1000)
                    _LOGGER.error(
                        "report_v2_publish_lock event=timeout waited_milliseconds=%d",
                        waited_milliseconds,
                    )
                    raise ReportPublishError(
                        "report v2 publish transaction lock timed out"
                    ) from exc
                time.sleep(min(poll_seconds, remaining))
    except OSError as exc:
        raise ReportPublishError(
            "report v2 publish transaction lock acquisition failed"
        ) from exc

    owner_path = lock_path / _OWNER_FILE_NAME
    try:
        descriptor = os.open(
            owner_path,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o600,
        )
        with os.fdopen(descriptor, "w", encoding="ascii", newline="") as stream:
            stream.write(owner_token)
            stream.flush()
            os.fsync(stream.fileno())
    except OSError as exc:
        # The directory remains as a fail-closed stale lock for manual recovery.
        raise ReportPublishError(
            "report v2 publish transaction lock initialization failed"
        ) from exc
    waited_milliseconds = int((time.monotonic() - started) * 1000)
    _emit_lock_receipt(
        "acquired",
        waited_milliseconds=waited_milliseconds,
    )
    return lock_path, owner_token, waited_milliseconds


def _release_report_v2_publish_lock(lock_path: Path, owner_token: str) -> None:
    owner_path = lock_path / _OWNER_FILE_NAME
    try:
        recorded_owner = owner_path.read_text(encoding="ascii")
    except (OSError, UnicodeError) as exc:
        raise ReportPublishError(
            "report v2 publish transaction lock ownership cannot be verified"
        ) from exc
    if not secrets.compare_digest(recorded_owner, owner_token):
        raise ReportPublishError(
            "report v2 publish transaction lock ownership mismatch"
        )

    try:
        owner_path.unlink()
        lock_path.rmdir()
    except OSError as exc:
        raise ReportPublishError(
            "report v2 publish transaction lock release failed"
        ) from exc


@contextmanager
def report_v2_publish_lock(
    publish_root: Path,
    *,
    wait_seconds: float = _DEFAULT_WAIT_SECONDS,
    poll_seconds: float = _DEFAULT_POLL_SECONDS,
) -> Iterator[dict[str, object]]:
    """Serialize one complete report-v2 transaction across processes."""
    if (
        isinstance(wait_seconds, bool)
        or not isinstance(wait_seconds, (int, float))
        or not 0 < float(wait_seconds) <= 600
        or isinstance(poll_seconds, bool)
        or not isinstance(poll_seconds, (int, float))
        or not 0 < float(poll_seconds) <= 1
    ):
        raise ReportPublishError("report v2 publish transaction lock wait is invalid")
    lock_path, owner_token, waited_milliseconds = _acquire_report_v2_publish_lock(
        publish_root,
        wait_seconds=float(wait_seconds),
        poll_seconds=float(poll_seconds),
    )
    receipt = {
        "scope": "report_v2_publish",
        "waited_milliseconds": waited_milliseconds,
        "acquired": True,
        "released": False,
    }
    body_error: BaseException | None = None
    try:
        yield receipt
    except BaseException as exc:
        body_error = exc
        raise
    finally:
        try:
            _release_report_v2_publish_lock(lock_path, owner_token)
            receipt["released"] = True
            _emit_lock_receipt(
                "released",
                waited_milliseconds=waited_milliseconds,
            )
        except ReportPublishError as release_error:
            _LOGGER.error(
                "report_v2_publish_lock event=release_failed waited_milliseconds=%d",
                waited_milliseconds,
            )
            if body_error is not None:
                raise ReportPublishError(
                    "report v2 publish failed and transaction lock release failed"
                ) from release_error
            raise
