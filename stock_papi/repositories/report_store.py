import hashlib
import hmac
import json
import re

from reporting.web import validate_report_index, validate_report_metadata


def _version(value):
    if value not in {"v1", "v2"}:
        raise ValueError("unsupported report store version")
    return value


def load_report_index(*, load_object, max_bytes, version="v1", market="TW"):
    version = _version(version)
    if market not in ("TW", "US"):
        raise ValueError("unsupported report store market")
    content = load_object(f"reports/{version}/index-{market}.json", max_bytes)
    return (
        None
        if content is None
        else validate_report_index(
            content,
            expected_version=int(version[1:]),
            expected_market=market,
        )
    )


def load_report_pdf(item, *, load_object, version="v1"):
    version = _version(version)
    content = load_object(f"reports/{version}/{item['pdf_path']}", item["pdf_size"])
    if (
        content is None
        or len(content) != item["pdf_size"]
        or not hmac.compare_digest(
            hashlib.sha256(content).hexdigest(), item["pdf_sha256"]
        )
    ):
        return None
    return content


def load_report_metadata(
    item, *, load_object, max_bytes=2 * 1024 * 1024, version="v1"
):
    version = _version(version)
    content = load_object(f"reports/{version}/{item['metadata']}", max_bytes)
    return (
        None
        if content is None
        else validate_report_metadata(
            content, item, expected_version=int(version[1:])
        )
    )


def load_report_metadata_by_sha(
    metadata_sha256, *, load_object, max_bytes=2 * 1024 * 1024
):
    if not isinstance(metadata_sha256, str) or re.fullmatch(
        r"[0-9a-f]{64}", metadata_sha256
    ) is None:
        return None
    content = load_object(
        f"reports/v2/metadata/{metadata_sha256}.json", max_bytes
    )
    if (
        not isinstance(content, bytes)
        or not hmac.compare_digest(hashlib.sha256(content).hexdigest(), metadata_sha256)
    ):
        return None
    try:
        document = json.loads(content)
        item = {
            key: document[key]
            for key in (
                "report_type",
                "source_market_date",
                "applicable_trading_date",
                "published_at",
                "data_as_of",
                "model_versions",
                "title",
                "summary",
                "content_sha256",
            )
        }
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        return None
    item["market"] = document.get("market") or "TW"
    item["metadata_sha256"] = metadata_sha256
    if document.get("product_mode") is not None:
        item["product_mode"] = document["product_mode"]
    return validate_report_metadata(content, item, expected_version=2)
