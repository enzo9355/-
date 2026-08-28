import re
import urllib.parse


_ALLOWED_PREFIXES = {
    "quant/v1/",
    "reports/v1/",
    "reports/v2/",
    "dashboard/v1/",
    "predictions/v1/",
    "previews/",
}


def get_allowed_object(
    object_name,
    max_bytes,
    allowed_prefix,
    *,
    bucket,
    enabled,
    token_provider,
    http_get,
):
    if (
        not enabled
        or not re.fullmatch(r"[a-z0-9][a-z0-9._-]{1,220}[a-z0-9]", bucket or "")
        or not isinstance(object_name, str)
        or not isinstance(allowed_prefix, str)
        or allowed_prefix not in _ALLOWED_PREFIXES
        or not object_name.startswith(allowed_prefix)
        or type(max_bytes) is not int
        or max_bytes < 1
    ):
        return None
    response = None
    try:
        response = http_get(
            "https://storage.googleapis.com/storage/v1/b/"
            f"{bucket}/o/{urllib.parse.quote(object_name, safe='')}?alt=media",
            headers={"Authorization": f"Bearer {token_provider()}"},
            timeout=5,
            stream=True,
        )
        if response.status_code != 200:
            return None
        content_length = response.headers.get("Content-Length")
        if content_length is not None and int(content_length) > max_bytes:
            return None
        content = bytearray()
        for chunk in response.iter_content(chunk_size=64 * 1024):
            content.extend(chunk)
            if len(content) > max_bytes:
                return None
        return bytes(content) if content else None
    except Exception:
        return None
    finally:
        if response is not None:
            response.close()
