"""可跨日續跑、固定 dataset identity 的 full-backtest worker。"""

import datetime
import hashlib
import json
import os
import re
import stat
from pathlib import Path

from stock_papi.batch.backtest_store import BacktestStoreError
from stock_papi.batch.runtime import (
    acquire_job_lock,
    job_namespace,
    yield_full_backtest_to_daily,
)


def _write_atomic(path, document):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    encoded = json.dumps(
        document, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    with temporary.open("wb") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _is_reparse_point(path):
    metadata = Path(path).lstat()
    return bool(
        getattr(metadata, "st_file_attributes", 0)
        & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
    ) or Path(path).is_symlink()


def _parse_utc_timestamp(value):
    if not isinstance(value, str) or not value.endswith("Z"):
        return None
    try:
        parsed = datetime.datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError:
        return None
    return parsed if parsed.utcoffset() == datetime.timedelta(0) else None


class FullBacktestWorker:
    def __init__(
        self,
        root,
        *,
        dataset_manifest,
        dataset_sha256,
        model_version,
        feature_schema_version,
        cutoff,
        items,
    ):
        manifest_pattern = (
            r"quant/v1/manifests/TW-[0-9]{8}T[0-9]{6}Z-[0-9a-f]{12}\.json"
        )
        normalized_items = tuple(str(item) for item in items)
        if (
            re.fullmatch(manifest_pattern, str(dataset_manifest)) is None
            or re.fullmatch(r"[0-9a-f]{64}", str(dataset_sha256)) is None
            or not isinstance(model_version, str)
            or not 1 <= len(model_version) <= 100
            or type(feature_schema_version) is not int
            or feature_schema_version < 1
            or type(cutoff) is not datetime.date
            or not normalized_items
            or len(normalized_items) != len(set(normalized_items))
            or not all(re.fullmatch(r"[A-Z0-9.-]{1,12}", item) for item in normalized_items)
        ):
            raise BacktestStoreError("full backtest worker input is invalid")
        self.root = Path(root)
        self.dataset_manifest = dataset_manifest
        self.dataset_sha256 = dataset_sha256
        self.model_version = model_version
        self.feature_schema_version = feature_schema_version
        self.cutoff = cutoff
        self.items = normalized_items
        self.items_sha256 = hashlib.sha256(
            json.dumps(normalized_items, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        self.checkpoint_path = job_namespace(root, "full_backtest").checkpoint

    def _identity(self):
        return {
            "dataset_manifest": self.dataset_manifest,
            "dataset_sha256": self.dataset_sha256,
            "model_version": self.model_version,
            "feature_schema_version": self.feature_schema_version,
            "cutoff": self.cutoff.isoformat(),
            "items_sha256": self.items_sha256,
            "item_count": len(self.items),
        }

    def verify_completed_checkpoint(self):
        """Return True only for a fully completed checkpoint of this exact source."""
        if not self.checkpoint_path.exists():
            return False
        try:
            checkpoint = json.loads(self.checkpoint_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, ValueError) as exc:
            raise BacktestStoreError("full backtest checkpoint is invalid") from exc
        identity = self._identity()
        next_index = checkpoint.get("next_index") if isinstance(checkpoint, dict) else None
        if (
            not isinstance(checkpoint, dict)
            or checkpoint.get("schema_version") != 1
            or checkpoint.get("job_type") != "full_backtest"
            or any(checkpoint.get(key) != value for key, value in identity.items())
            or type(next_index) is not int
            or not 0 <= next_index <= len(self.items)
            or checkpoint.get("completed_items") != list(self.items[:next_index])
            or checkpoint.get("status") not in {
                "running", "yielded", "failed", "completed"
            }
        ):
            raise BacktestStoreError("full backtest completed checkpoint identity mismatch")
        if checkpoint["status"] != "completed":
            return False
        if next_index != len(self.items):
            raise BacktestStoreError("full backtest completed checkpoint identity mismatch")
        return True

    def _read_archivable_stale_completed_checkpoint(self):
        if not self.checkpoint_path.exists():
            return None
        self._assert_checkpoint_paths_are_plain()
        try:
            raw = self.checkpoint_path.read_bytes()
            checkpoint = json.loads(raw.decode("utf-8"))
        except (OSError, UnicodeError, ValueError) as exc:
            raise BacktestStoreError("full backtest checkpoint is invalid") from exc
        if not isinstance(checkpoint, dict):
            raise BacktestStoreError("full backtest checkpoint is invalid")
        if all(checkpoint.get(key) == value for key, value in self._identity().items()):
            return None
        completed_items = checkpoint.get("completed_items")
        item_count = checkpoint.get("item_count")
        next_index = checkpoint.get("next_index")
        dataset_manifest = checkpoint.get("dataset_manifest")
        dataset_sha256 = checkpoint.get("dataset_sha256")
        model_version = checkpoint.get("model_version")
        feature_schema_version = checkpoint.get("feature_schema_version")
        cutoff = checkpoint.get("cutoff")
        items_sha256 = checkpoint.get("items_sha256")
        allowed_keys = {
            "schema_version",
            "job_type",
            "run_id",
            "dataset_manifest",
            "dataset_sha256",
            "model_version",
            "feature_schema_version",
            "cutoff",
            "items_sha256",
            "item_count",
            "next_index",
            "completed_items",
            "status",
            "started_at",
            "updated_at",
            "last_error_type",
        }
        manifest_match = re.fullmatch(
            r"quant/v1/manifests/TW-[0-9]{8}T[0-9]{6}Z-([0-9a-f]{12})\.json",
            str(dataset_manifest),
        )
        try:
            parsed_cutoff = datetime.date.fromisoformat(str(cutoff))
        except ValueError:
            parsed_cutoff = None
        computed_items_sha256 = None
        if isinstance(completed_items, list):
            computed_items_sha256 = hashlib.sha256(
                json.dumps(tuple(completed_items), separators=(",", ":")).encode("utf-8")
            ).hexdigest()
        if (
            checkpoint.get("schema_version") != 1
            or checkpoint.get("job_type") != "full_backtest"
            or checkpoint.get("status") != "completed"
            or not set(checkpoint).issubset(allowed_keys)
            or manifest_match is None
            or re.fullmatch(r"[0-9a-f]{64}", str(dataset_sha256)) is None
            or manifest_match.group(1) != str(dataset_sha256)[:12]
            or not isinstance(model_version, str)
            or not 1 <= len(model_version) <= 100
            or type(feature_schema_version) is not int
            or feature_schema_version < 1
            or parsed_cutoff is None
            or parsed_cutoff.isoformat() != cutoff
            or checkpoint.get("run_id")
            != f"{parsed_cutoff.strftime('%Y%m%d')}T000000Z-{str(dataset_sha256)[:8]}"
            or _parse_utc_timestamp(checkpoint.get("started_at")) is None
            or _parse_utc_timestamp(checkpoint.get("updated_at")) is None
            or _parse_utc_timestamp(checkpoint.get("updated_at"))
            < _parse_utc_timestamp(checkpoint.get("started_at"))
            or (
                "last_error_type" in checkpoint
                and (
                    not isinstance(checkpoint["last_error_type"], str)
                    or re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]{0,99}", checkpoint["last_error_type"])
                    is None
                )
            )
            or re.fullmatch(r"[0-9a-f]{64}", str(items_sha256)) is None
            or type(item_count) is not int
            or item_count < 1
            or type(next_index) is not int
            or next_index != item_count
            or not isinstance(completed_items, list)
            or len(completed_items) != item_count
            or len(completed_items) != len(set(completed_items))
            or not all(
                isinstance(item, str)
                and re.fullmatch(r"[A-Z0-9.-]{1,12}", item)
                for item in completed_items
            )
            or computed_items_sha256 != items_sha256
        ):
            raise BacktestStoreError(
                "full backtest stale completed checkpoint is not self-consistent"
            )
        return raw

    def has_archivable_stale_completed_checkpoint(self):
        """Return True only for a self-consistent completed prior-source checkpoint."""
        return self._read_archivable_stale_completed_checkpoint() is not None

    def _assert_checkpoint_paths_are_plain(self):
        root = Path(os.path.abspath(self.root))
        checkpoint = Path(os.path.abspath(self.checkpoint_path))
        try:
            relative = checkpoint.relative_to(root)
        except ValueError as exc:
            raise BacktestStoreError("full backtest checkpoint escaped data root") from exc
        current = root
        candidates = [current]
        for part in relative.parts:
            current = current / part
            candidates.append(current)
        for candidate in candidates:
            if candidate.exists() and _is_reparse_point(candidate):
                raise BacktestStoreError(
                    "full backtest checkpoint path contains a reparse point"
                )

    @staticmethod
    def _write_archive_exclusive(path, raw):
        try:
            descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        except FileExistsError:
            if _is_reparse_point(path):
                raise BacktestStoreError(
                    "full backtest checkpoint archive target is a reparse point"
                )
            if Path(path).read_bytes() != raw:
                raise BacktestStoreError("full backtest checkpoint archive collision")
            return
        try:
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(raw)
                stream.flush()
                os.fsync(stream.fileno())
        except BaseException:
            try:
                Path(path).unlink()
            except OSError:
                pass
            raise

    def _archive_stale_completed_checkpoint(self):
        raw = self._read_archivable_stale_completed_checkpoint()
        if raw is None:
            return False
        archive_dir = self.checkpoint_path.parent / "archive"
        archive_dir.mkdir(parents=True, exist_ok=True)
        self._assert_checkpoint_paths_are_plain()
        if _is_reparse_point(archive_dir):
            raise BacktestStoreError(
                "full backtest checkpoint archive is a reparse point"
            )
        digest = hashlib.sha256(raw).hexdigest()
        archive_path = archive_dir / f"completed-{digest}.json"
        retiring_path = archive_dir / f".retiring-{digest}.json"
        if retiring_path.exists():
            raise BacktestStoreError("full backtest checkpoint retirement is pending")
        self._write_archive_exclusive(archive_path, raw)
        self.checkpoint_path.rename(retiring_path)
        retired_raw = retiring_path.read_bytes()
        if retired_raw != raw:
            if not self.checkpoint_path.exists():
                retiring_path.rename(self.checkpoint_path)
            else:
                rejected = archive_dir / (
                    f"rejected-{hashlib.sha256(retired_raw).hexdigest()}.json"
                )
                if rejected.exists():
                    if rejected.read_bytes() != retired_raw:
                        raise BacktestStoreError(
                            "full backtest checkpoint rejected archive collision"
                        )
                    retiring_path.unlink()
                else:
                    retiring_path.rename(rejected)
            raise BacktestStoreError("full backtest checkpoint changed during archive")
        retiring_path.unlink()
        return True

    def _load_or_create(self, checked_at):
        identity = self._identity()
        if self.checkpoint_path.exists():
            try:
                checkpoint = json.loads(
                    self.checkpoint_path.read_text(encoding="utf-8")
                )
            except (OSError, ValueError) as exc:
                raise BacktestStoreError("full backtest checkpoint is invalid") from exc
            if (
                checkpoint.get("schema_version") != 1
                or checkpoint.get("job_type") != "full_backtest"
                or any(checkpoint.get(key) != value for key, value in identity.items())
                or type(checkpoint.get("next_index")) is not int
                or not 0 <= checkpoint["next_index"] <= len(self.items)
            ):
                raise BacktestStoreError("full backtest checkpoint identity mismatch")
            return checkpoint
        checkpoint = {
            "schema_version": 1,
            "job_type": "full_backtest",
            "run_id": (
                f"{self.cutoff.strftime('%Y%m%d')}T000000Z-"
                f"{self.dataset_sha256[:8]}"
            ),
            **identity,
            "next_index": 0,
            "completed_items": [],
            "status": "running",
            "started_at": checked_at.astimezone(datetime.timezone.utc)
            .isoformat()
            .replace("+00:00", "Z"),
            "updated_at": checked_at.astimezone(datetime.timezone.utc)
            .isoformat()
            .replace("+00:00", "Z"),
        }
        _write_atomic(self.checkpoint_path, checkpoint)
        return checkpoint

    def run(
        self,
        run_item,
        *,
        max_items=None,
        now=None,
        archive_stale_completed=False,
    ):
        if not callable(run_item):
            raise TypeError("run_item must be callable")
        if max_items is not None and (type(max_items) is not int or max_items < 1):
            raise ValueError("max_items must be a positive integer")
        if type(archive_stale_completed) is not bool:
            raise ValueError("archive_stale_completed must be boolean")
        checked_at = now or datetime.datetime.now(datetime.timezone.utc)
        if checked_at.tzinfo is None or checked_at.utcoffset() is None:
            raise ValueError("now must be timezone-aware")

        with acquire_job_lock(
            self.root,
            "full_backtest",
            self.cutoff,
            now=checked_at,
        ):
            if archive_stale_completed:
                self._archive_stale_completed_checkpoint()
            checkpoint = self._load_or_create(checked_at)

            def save(reason=None):
                checkpoint["updated_at"] = checked_at.astimezone(
                    datetime.timezone.utc
                ).isoformat().replace("+00:00", "Z")
                if reason is not None:
                    checkpoint["status"] = "yielded"
                    checkpoint["yield_reason"] = reason
                _write_atomic(self.checkpoint_path, checkpoint)

            processed = 0
            while checkpoint["next_index"] < len(self.items):
                if max_items is not None and processed >= max_items:
                    break
                if yield_full_backtest_to_daily(
                    self.root, boundary="symbol", save_checkpoint=save
                ):
                    return dict(checkpoint)
                item = self.items[checkpoint["next_index"]]
                try:
                    run_item(item)
                except Exception as exc:
                    checkpoint["status"] = "failed"
                    checkpoint["last_error_type"] = type(exc).__name__
                    save()
                    raise
                checkpoint["completed_items"].append(item)
                checkpoint["next_index"] += 1
                checkpoint["status"] = "running"
                checkpoint.pop("yield_reason", None)
                save()
                processed += 1
            if checkpoint["next_index"] == len(self.items):
                checkpoint["status"] = "completed"
                save()
            return dict(checkpoint)
