"""Verified reader for immutable five-session prediction products."""

import datetime
import hashlib
import hmac
import json
import re
import time

from stock_papi.batch.prediction_products import validate_prediction_product


MAX_PREDICTION_BYTES = 5_000_000
PREDICTION_CACHE = {}


def load_prediction_snapshot(market, today=None, *, load_object, cache=PREDICTION_CACHE):
    if market not in {"TW", "US"}:
        return None
    now = time.time()
    cached = cache.get(market)
    if cached and now - cached[1] < 300:
        return cached[0]
    pointer_bytes = load_object(f"predictions/v1/latest-{market}.json", 100_000)
    if pointer_bytes is None:
        return None
    try:
        pointer = json.loads(pointer_bytes.decode("utf-8"))
        digest = str(pointer["sha256"])
        path = str(pointer["path"])
        size = pointer["size"]
        as_of = datetime.date.fromisoformat(str(pointer["as_of"]))
        age = (today or datetime.date.today()) - as_of
        schema_version = pointer.get("schema_version")
        research = schema_version == 2 and pointer.get("validation_mode") == "research"
        promoted = schema_version == 1 and isinstance(pointer.get("backtest_sha256"), str)
        if (
            not (research or promoted)
            or pointer.get("kind") != "absorb-five-session-predictions-pointer"
            or pointer.get("market") != market
            or re.fullmatch(r"[0-9a-f]{64}", digest) is None
            or path != f"objects/{digest}.json"
            or type(size) is not int
            or not 0 < size <= MAX_PREDICTION_BYTES
            or not 0 <= age.days <= 7
        ):
            return None
        body = load_object(f"predictions/v1/{path}", size)
        if body is None or len(body) != size or not hmac.compare_digest(hashlib.sha256(body).hexdigest(), digest):
            return None
        document = json.loads(body.decode("utf-8"))
        validate_prediction_product(document)
        fields = ["market", "as_of", "source_manifest", "source_manifest_sha256"]
        fields.append("validation_mode" if research else "backtest_sha256")
        if any(
            document.get(field) != pointer.get(field)
            for field in fields
        ):
            return None
    except (KeyError, TypeError, UnicodeError, ValueError):
        return None
    cache[market] = (document, now)
    return document
