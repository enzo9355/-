"""Build local, immutable five-session prediction products for TW or US."""

import argparse
import datetime
import hashlib
import json
import math
import os
import sys
from pathlib import Path

from stock_papi.batch.prediction_products import (
    INDEX_SYMBOLS,
    build_prediction_product,
    build_research_prediction_product,
)
from stock_papi.quant.constants import MODEL_FEATURES


MAX_PRODUCT_BYTES = 5_000_000


def _canonical(document):
    return json.dumps(
        document,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _write_exclusive(path, content):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError:
        if path.read_bytes() != content:
            raise ValueError("immutable prediction object conflict")
        return
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(content)
        stream.flush()
        os.fsync(stream.fileno())


def _write_atomic(path, content):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        stream.write(content)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _frame_rows(daily, pipeline, as_of):
    frame = pipeline.pd.DataFrame(daily)
    if frame.empty or "Date" not in frame:
        return None
    frame["Date"] = pipeline.pd.to_datetime(frame["Date"], errors="coerce")
    frame = frame.dropna(subset=["Date"]).set_index("Date").sort_index()
    frame = frame[frame.index.date <= as_of]
    if frame.empty:
        return None
    if any(
        column not in frame or pipeline.pd.isna(frame.iloc[-1][column])
        for column in MODEL_FEATURES
    ):
        frame = pipeline.calc_all(frame.drop(
            columns=["AI_P", "AI_PRED_RET_5", "AI_PRED_PRICE_5"],
            errors="ignore",
        ))
    if frame.empty or frame.index[-1].date() != as_of:
        return None
    inference = pipeline.run_latest_inference(frame)
    if not isinstance(inference, dict):
        return None
    return inference, json.loads(
        frame.reset_index().to_json(
            orient="records", date_format="iso", date_unit="ms", double_precision=15
        )
    )


def _merge_daily(historical, current):
    rows = {}
    for row in list(historical or []) + list(current or []):
        if not isinstance(row, dict):
            raise ValueError("prediction history row is invalid")
        try:
            day = datetime.date.fromisoformat(str(row.get("Date"))[:10])
        except (TypeError, ValueError) as exc:
            raise ValueError("prediction history date is invalid") from exc
        rows[day] = dict(row)
    return [rows[day] for day in sorted(rows)]


def _usable_inference(inferred):
    if not isinstance(inferred, tuple) or len(inferred) != 2 or not inferred[1]:
        return False
    latest = inferred[1][-1]
    try:
        current = float(latest["Close"])
        probability = float(latest["AI_P"])
        predicted_return = float(latest["AI_PRED_RET_5"])
        predicted_price = float(latest["AI_PRED_PRICE_5"])
    except (KeyError, TypeError, ValueError):
        return False
    return (
        all(math.isfinite(value) for value in (
            current, probability, predicted_return, predicted_price
        ))
        and current > 0
        and 0 <= probability <= 100
        and predicted_price > 0
        and math.isclose(
            predicted_price, current * (1 + predicted_return), rel_tol=1e-9
        )
    )


def _index_frame(pipeline, market, symbol, as_of):
    start = (as_of - datetime.timedelta(days=1095)).isoformat()
    end = (as_of + datetime.timedelta(days=1)).isoformat()
    ticker = "^TWII" if symbol == "TAIEX" else symbol
    benchmark = "^TWII" if market == "TW" else "^GSPC"
    etf = "0050.TW" if market == "TW" else "SPY"
    price = pipeline.fetch_yfinance_price_history(ticker, start, end)
    if price is None or price.empty:
        return None
    price = pipeline.add_price_quality_features(price)
    market_frame = pipeline.fetch_yfinance_price_history(benchmark, start, end)
    etf_frame = pipeline.fetch_yfinance_price_history(etf, start, end)
    price = pipeline.add_market_context_features(price, market_frame, etf_frame)
    price = pipeline.add_option_context_features(
        price, *pipeline.fetch_option_context_history(start, end)
    )
    return pipeline._clean_df(pipeline.merge_chip_data(price))


def _us_index_frame(pipeline, symbol, as_of):
    return _index_frame(pipeline, "US", symbol, as_of)


def _prepare_snapshots(source, backtest):
    from local_quant import load_stock_pipeline

    pipeline = load_stock_pipeline(source.root)
    market = source.manifest.market
    as_of = source.manifest.market_as_of
    expected_model = backtest["model_version"]
    feature_schema = backtest["feature_schema_version"]
    snapshots = {}
    for stock in source.stocks:
        if stock.observation_kind != "regular_price" or stock.as_of != as_of:
            continue
        inferred = _frame_rows(stock.daily, pipeline, as_of)
        if inferred is None or inferred[0].get("model_version") != expected_model:
            continue
        snapshots[stock.symbol] = {
            "schema_version": 2,
            "market": market,
            "symbol": stock.symbol,
            "as_of": as_of.isoformat(),
            "model_version": expected_model,
            "feature_schema_version": feature_schema,
            "daily": inferred[1],
        }
    for symbol in INDEX_SYMBOLS[market]:
        frame = (
            pipeline.get_data("TAIEX", 1095)
            if market == "TW"
            else _us_index_frame(pipeline, symbol, as_of)
        )
        if frame is None or frame.empty:
            raise ValueError(f"verified index history is unavailable: {symbol}")
        frame = frame[frame.index.date <= as_of]
        calculated = pipeline.calc_all(frame)
        if calculated is None or calculated.empty:
            raise ValueError(f"index features are unavailable: {symbol}")
        inferred = _frame_rows(
            json.loads(
                calculated.reset_index().to_json(
                    orient="records", date_format="iso", date_unit="ms", double_precision=15
                )
            ),
            pipeline,
            as_of,
        )
        if inferred is None or inferred[0].get("model_version") != expected_model:
            raise ValueError(f"index prediction is unavailable: {symbol}")
        snapshots[symbol] = {
            "schema_version": 2,
            "market": market,
            "symbol": symbol,
            "as_of": as_of.isoformat(),
            "model_version": expected_model,
            "feature_schema_version": feature_schema,
            "daily": inferred[1],
        }
    return snapshots


def _prepare_research_snapshots(source, history_root):
    from local_quant import load_stock_pipeline
    from reporting.source_loader import load_report_source
    from stock_papi.quant.model import FEATURE_SCHEMA_VERSION, MODEL_VERSION

    pipeline = load_stock_pipeline(source.root)
    market = source.manifest.market
    as_of = source.manifest.market_as_of
    history = load_report_source(history_root, market) if market == "TW" else None
    historical = {
        stock.symbol: stock.daily
        for stock in history.stocks
        if stock.observation_kind == "regular_price"
    } if history is not None else {}
    snapshots = {}
    for index, stock in enumerate(source.stocks, 1):
        if stock.observation_kind != "regular_price" or stock.as_of != as_of:
            continue
        inferred = _frame_rows(
            _merge_daily(historical.get(stock.symbol), stock.daily), pipeline, as_of
        )
        if (
            not _usable_inference(inferred)
            or inferred[0].get("model_version") != MODEL_VERSION
        ):
            continue
        snapshots[stock.symbol] = {
            "schema_version": 2,
            "market": market,
            "symbol": stock.symbol,
            "as_of": as_of.isoformat(),
            "model_version": MODEL_VERSION,
            "feature_schema_version": FEATURE_SCHEMA_VERSION,
            "daily": inferred[1],
        }
        if index % 100 == 0:
            print(f"{market} research inference {index}/{len(source.stocks)}", file=sys.stderr)
    for symbol in INDEX_SYMBOLS[market]:
        frame = _index_frame(pipeline, market, symbol, as_of)
        if frame is None or frame.empty:
            raise ValueError(f"verified index history is unavailable: {symbol}")
        calculated = pipeline.calc_all(frame[frame.index.date <= as_of])
        inferred = _frame_rows(
            json.loads(calculated.reset_index().to_json(
                orient="records", date_format="iso", date_unit="ms", double_precision=15
            )),
            pipeline,
            as_of,
        ) if calculated is not None and not calculated.empty else None
        if inferred is None or inferred[0].get("model_version") != MODEL_VERSION:
            raise ValueError(f"index prediction is unavailable: {symbol}")
        snapshots[symbol] = {
            "schema_version": 2,
            "market": market,
            "symbol": symbol,
            "as_of": as_of.isoformat(),
            "model_version": MODEL_VERSION,
            "feature_schema_version": FEATURE_SCHEMA_VERSION,
            "daily": inferred[1],
        }
    return snapshots


def _fifth_session(root, market, as_of):
    from stock_papi.batch.calendar import TradingCalendar, TradingCalendarSet

    documents = []
    for year in (as_of.year, as_of.year + 1):
        path = Path(root) / "publish" / "calendars" / "v1" / f"{market}-{year}.json"
        if path.is_file():
            document = json.loads(path.read_text(encoding="utf-8"))
            calendar = TradingCalendar.from_document(document)
            if calendar.market != market:
                raise ValueError("calendar market mismatch")
            documents.append(document)
    return TradingCalendarSet.from_documents(documents).session_offset(as_of, 5)


def build(
    root,
    market,
    *,
    mode="promoted",
    history_root=Path(r"D:\StockPapiData"),
    now=None,
    load_source=None,
    load_backtest=None,
    prepare_snapshots=None,
    fifth_session=None,
):
    if market not in INDEX_SYMBOLS or mode not in {"promoted", "research"}:
        raise ValueError("unsupported prediction market")
    root = Path(root)
    if load_source is None:
        from reporting.source_loader import load_report_source

        load_source = load_report_source
    if load_backtest is None:
        from stock_papi.batch.backtest_store import BacktestStore

        load_backtest = lambda selected_root, selected_market: BacktestStore(
            selected_root, selected_market
        ).load_latest()
    source = load_source(root, market)
    manifest = source.manifest
    if (
        manifest.schema_version != 4
        or manifest.market != market
        or not manifest.manifest_path.startswith(f"manifests/{market}-")
    ):
        raise ValueError("prediction source manifest is invalid")
    backtest = load_backtest(root, market) if mode == "promoted" else None
    if mode == "promoted" and not isinstance(backtest, dict):
        raise ValueError("promoted prediction inputs are unavailable")
    source.root = root
    snapshots = (
        prepare_snapshots(source, backtest)
        if prepare_snapshots is not None
        else _prepare_research_snapshots(source, Path(history_root))
        if mode == "research"
        else _prepare_snapshots(source, backtest)
    )
    if not snapshots:
        raise ValueError("promoted prediction inputs are unavailable")
    as_of = manifest.market_as_of
    source_symbols = {
        stock.symbol
        for stock in source.stocks
        if stock.observation_kind == "regular_price" and stock.as_of == as_of
    }
    symbols = source_symbols | set(INDEX_SYMBOLS[market]) if mode == "research" else set(snapshots)
    quant_manifest = {
        "schema_version": 4,
        "market": market,
        "observation_as_of": as_of.isoformat(),
        "source_manifest": f"quant/v1/{manifest.manifest_path}",
        "source_manifest_sha256": manifest.manifest_sha256,
        "symbols": {
            symbol: {"as_of": as_of.isoformat()} for symbol in symbols
        },
    }
    generated_at = now or datetime.datetime.now(datetime.timezone.utc)
    target = (fifth_session or _fifth_session)(root, market, as_of)
    common = {
        "next_session": lambda _market, _as_of, _count: target,
        "generated_at": generated_at,
    }
    product = (
        build_prediction_product(market, quant_manifest, snapshots, backtest, **common)
        if mode == "promoted"
        else build_research_prediction_product(market, quant_manifest, snapshots, **common)
    )
    content = _canonical(product)
    if not 0 < len(content) <= MAX_PRODUCT_BYTES:
        raise ValueError("prediction product size is invalid")
    digest = hashlib.sha256(content).hexdigest()
    publish = root / "publish" / "predictions" / "v1"
    object_path = publish / "objects" / f"{digest}.json"
    pointer_path = publish / f"latest-{market}.json"
    _write_exclusive(object_path, content)
    pointer = {
        "schema_version": product["schema_version"],
        "kind": "absorb-five-session-predictions-pointer",
        "market": market,
        "as_of": product["as_of"],
        "path": f"objects/{digest}.json",
        "sha256": digest,
        "size": len(content),
        "source_manifest": product["source_manifest"],
        "source_manifest_sha256": product["source_manifest_sha256"],
    }
    if mode == "promoted":
        pointer["backtest_sha256"] = product["backtest_sha256"]
    else:
        pointer["validation_mode"] = "research"
    _write_atomic(pointer_path, _canonical(pointer))
    return {
        "market": market,
        "as_of": product["as_of"],
        "object_path": str(object_path),
        "pointer_path": str(pointer_path),
        "sha256": digest,
        "entity_count": len(product["entities"]),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description="Build ABSORB prediction products")
    parser.add_argument("--root", type=Path, default=Path(r"D:\AbsorbData"))
    parser.add_argument("--market", choices=("TW", "US"), required=True)
    parser.add_argument("--mode", choices=("promoted", "research"), default="promoted")
    parser.add_argument("--history-root", type=Path, default=Path(r"D:\StockPapiData"))
    args = parser.parse_args(argv)
    print(json.dumps(build(
        args.root, args.market, mode=args.mode, history_root=args.history_root
    ), ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
