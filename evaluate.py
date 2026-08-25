import argparse
from pathlib import Path

import pandas as pd

from crypto.evaluation import metric_tables, validate_predictions
from crypto.evaluation_plots import render_bundle, require_volatility_baseline


def main(argv=None):
    parser = argparse.ArgumentParser(description="Render daily OOS evaluation charts.")
    parser.add_argument("predictions", nargs="+", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    frame = pd.concat([pd.read_parquet(path) for path in args.predictions], ignore_index=True)
    validate_predictions(frame)
    require_volatility_baseline(frame)
    args.out.mkdir(parents=True, exist_ok=False)
    for name, table in metric_tables(frame).items():
        table.to_csv(args.out / f"metrics_{name}.csv")
    render_bundle(frame, args.out)


if __name__ == "__main__":
    main()
