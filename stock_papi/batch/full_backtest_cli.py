"""以固定 immutable manifest 執行可續跑的逐檔完整回測。"""

import argparse
import datetime
import json
import os
from pathlib import Path


def _write_atomic(path, document):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    encoded = json.dumps(
        document,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    with temporary.open("wb") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def main(argv=None):
    parser = argparse.ArgumentParser(description="ABSORB resumable full backtest")
    parser.add_argument("--root", type=Path, default=Path(r"D:\AbsorbData"))
    parser.add_argument("--market", choices=("TW", "US"), default="TW")
    parser.add_argument("--max-items", type=int, default=25)
    parser.add_argument("--verify-completion", action="store_true")
    args = parser.parse_args(argv)

    from reporting.source_loader import load_report_source
    from stock_papi.batch.backtest_worker import FullBacktestWorker
    from stock_papi.batch.runtime import job_namespace
    from stock_papi.batch.status import PipelineStatusWriter

    source = load_report_source(args.root, market=args.market)
    manifest = source.manifest
    versions = {stock.model_version for stock in source.stocks}
    if len(versions) != 1:
        raise ValueError("full backtest requires one source model version")
    items = tuple(stock.symbol for stock in source.stocks)
    from stock_papi.quant.model import FEATURE_SCHEMA_VERSION, MODEL_VERSION

    model_version = (
        MODEL_VERSION
        if getattr(manifest, "schema_version", 2) in (3, 4)
        else next(iter(versions))
    )

    worker = FullBacktestWorker(
        args.root,
        market=args.market,
        dataset_manifest=f"quant/v1/{manifest.manifest_path}",
        dataset_sha256=manifest.manifest_sha256,
        model_version=model_version,
        feature_schema_version=FEATURE_SCHEMA_VERSION,
        cutoff=manifest.market_as_of,
        items=items,
    )
    if worker.verify_completed_checkpoint():
        print("full backtest checkpoint is already completed; skipping execution")
        return 0
    if args.verify_completion:
        print("full backtest checkpoint is not completed for the current source")
        return 3

    from local_quant import load_stock_pipeline
    pipeline = load_stock_pipeline(args.root)
    job_type = "full_backtest" if args.market == "TW" else "full_backtest_us"
    namespace = job_namespace(args.root, job_type)
    result_root = namespace.output / manifest.manifest_sha256 / "symbols"
    writer = PipelineStatusWriter(
        args.root,
        job_type=job_type,
        run_id=f"{manifest.market_as_of.strftime('%Y%m%d')}T000000Z-{manifest.manifest_sha256[:8]}",
        target_date=manifest.market_as_of,
    )

    def run_item(symbol):
        result_path = result_root / f"{symbol}.json"
        if result_path.exists():
            existing = json.loads(result_path.read_text(encoding="utf-8"))
            if (
                existing.get("dataset_sha256") == manifest.manifest_sha256
                and existing.get("symbol") == symbol
                and existing.get("model_version") == model_version
            ):
                return
            raise ValueError("full backtest result identity mismatch")
        frame = pipeline.get_data(symbol, 730, as_of=manifest.market_as_of)
        if frame is None or frame.empty:
            raise ValueError("price history is unavailable")
        frame = pipeline.calc_all(frame)
        if frame is None or frame.empty:
            raise ValueError("calculated history is unavailable")
        backtest = pipeline.run_ai_engine(frame, include_oos=True)
        if not isinstance(backtest, dict):
            raise ValueError("full backtest result is unavailable")
        oos_predictions = backtest.pop("oos_predictions", None)
        if (
            not isinstance(oos_predictions, list)
            or len(oos_predictions) < 30
            or backtest.get("five_session_gap") is not True
            or type(backtest.get("fold_count")) is not int
            or backtest["fold_count"] < 1
        ):
            raise ValueError("full backtest OOS evidence is unavailable")
        _write_atomic(
            result_path,
            {
                "schema_version": 1,
                "market": args.market,
                "symbol": symbol,
                "cutoff": manifest.market_as_of.isoformat(),
                "dataset_manifest": f"quant/v1/{manifest.manifest_path}",
                "dataset_sha256": manifest.manifest_sha256,
                "model_version": model_version,
                "generated_at": datetime.datetime.now(datetime.timezone.utc)
                .isoformat()
                .replace("+00:00", "Z"),
                "backtest": backtest,
                "oos_predictions": oos_predictions,
            },
        )

    writer.record("backtest", details={"processed": 0}, now=datetime.datetime.now(datetime.timezone.utc))
    try:
        result = worker.run(run_item, max_items=args.max_items)
    except Exception as exc:
        writer.record(
            "failed",
            details={"processed": 0},
            error=exc,
            now=datetime.datetime.now(datetime.timezone.utc),
        )
        raise
    final_stage = result["status"] if result["status"] in {"completed", "yielded"} else "backtest"
    writer.record(
        final_stage,
        details={"processed": result["next_index"]},
        now=datetime.datetime.now(datetime.timezone.utc),
    )
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
