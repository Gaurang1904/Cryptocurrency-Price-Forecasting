"""Backtest the neural family: DLinear vs LSTM. python -m neural.run

Both sequence models are scored on the same origins as tree/ and linear/.
"""

import argparse
from pathlib import Path

from neural.core import backtest, report_and_log
from neural.dlinear import build as dlinear
from neural.lstm import build as lstm

BUILDERS = {"dlinear": dlinear, "lstm": lstm}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Run the daily neural backtest.")
    parser.add_argument("--run-id", help="Unique run label; default is a UUID.")
    parser.add_argument(
        "--output-root", type=Path, default=Path("artifacts/evaluation"),
        help="Parent directory for the non-overwriting run bundle.",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    res, vol_err = backtest(
        BUILDERS, run_id=args.run_id, output_root=args.output_root
    )
    print(f"wrote OOS bundle to {res.attrs['output_dir']}")
    report_and_log(res, vol_err, "neural")


if __name__ == "__main__":
    main()
