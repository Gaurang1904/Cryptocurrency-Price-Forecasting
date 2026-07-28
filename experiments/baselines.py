"""Phase 0: what any model must beat. flat / drift / seasonal, walk-forward."""

import pandas as pd

from crypto.backtest import H, _self_check, log, report, walk_forward
from crypto.data import OHLCV_OUT

if __name__ == "__main__":
    _self_check()
    df = pd.read_parquet(OHLCV_OUT)
    res = walk_forward(df)
    summary, per_h = report(res)

    print(f"{len(res) // (H * 3)} origins, {df.asset.nunique()} assets, h=1..{H}\n")
    print("BASELINES (lower MAE/MAPE/MASE better, DirAcc vs 50%)")
    print(summary.round(2).to_string())
    print("\nMAPE % by horizon day")
    print(per_h.round(2).to_string())
    log("baselines", summary, origins=len(res) // (H * 3), assets=df.asset.nunique())
