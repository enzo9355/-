import hashlib
import hmac

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
        else validate_report_index(content, expected_version=int(version[1:]))
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
