"""Command-line entry point for immutable full-backtest candidate generation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from stock_papi.batch.backtest_candidate import build_candidate


def main(argv=None):
    parser = argparse.ArgumentParser(description="Build an ABSORB full-backtest candidate")
    parser.add_argument("--root", type=Path, default=Path(r"D:\AbsorbData"))
    parser.add_argument("--market", choices=("TW", "US"), default="TW")
    parser.add_argument("--git-sha", required=True)
    args = parser.parse_args(argv)
    candidate = build_candidate(args.root, market=args.market, git_sha=args.git_sha)
    print(json.dumps(candidate, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
