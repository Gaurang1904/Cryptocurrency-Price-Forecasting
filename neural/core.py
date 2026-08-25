"""Shared neural plumbing: windowing, scaling, training loop, backtest.

Neural models here are sequence-native: they read a lookback window of raw daily
signals and learn their own features, rather than consuming the engineered tree
features. Everything downstream - calibration z-table, score(), plots - is the
same as tree/ and linear/, so pinball and coverage compare directly.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from numpy.lib.stride_tricks import sliding_window_view

from crypto.backtest import H, run_folds, score, coverage_by_h, log
from crypto.features import build
from crypto.evaluation import reserve_run_dir, save_predictions
from crypto.model import bands, calibrate, clip_sigma, split_calibration

LOOKBACK = 30
CHANNELS = ["r", "ar", "rng", "dv"]  # log-ret, |log-ret|, range, log-volume change
EPOCHS, PATIENCE, LR = 200, 15, 1e-3
SEED = 42
DRIVER = ["vol_21d"]  # minimal non-NaN col to drive run_folds origin selection


def channel_windows(df, lookback=LOOKBACK):
    """Per-(asset, date): the preceding `lookback` days of CHANNELS, causal.

    Returns X (n, lookback, C) and an index frame (asset, date) aligned row-wise.
    """
    Xs, idx = [], []
    for asset, g in df.groupby("asset", sort=False):
        g = g.sort_values("date")
        r = np.log(g.close).diff()
        chan = np.column_stack([
            r, r.abs(), (g.high - g.low) / g.close, np.log(g.volume).diff(),
        ])
        chan = np.nan_to_num(chan)
        if len(chan) <= lookback:
            continue
        w = sliding_window_view(chan, lookback, axis=0)      # (T-L+1, C, L)
        w = np.moveaxis(w, 1, 2)                              # (T-L+1, L, C)
        Xs.append(w)
        idx.append(pd.DataFrame({"asset": asset, "date": g.date.to_numpy()[lookback - 1:]}))
    return np.concatenate(Xs), pd.concat(idx, ignore_index=True)


def _lookup(win, index, rows):
    """Windows for the given rows, plus a mask of which rows had one."""
    key = index.reset_index().set_index(["asset", "date"])["index"]
    want = pd.MultiIndex.from_frame(rows[["asset", "date"]])
    pos = key.reindex(want)
    ok = pos.notna().to_numpy()
    return win[pos[ok].astype(int).to_numpy()], ok


def _scale(x, mu, sd):
    return (x - mu) / sd


def train_net(model, Xtr, Ytr, Xva, Yva):
    """Adam + early stopping on validation MSE. Restores best weights."""
    torch.manual_seed(SEED)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    lossf = torch.nn.MSELoss()
    Xtr, Ytr, Xva, Yva = map(lambda a: torch.tensor(a, dtype=torch.float32), (Xtr, Ytr, Xva, Yva))

    best, best_state, wait = float("inf"), None, 0
    for _ in range(EPOCHS):
        model.train()
        for i in range(0, len(Xtr), 256):
            opt.zero_grad()
            lossf(model(Xtr[i:i + 256]), Ytr[i:i + 256]).backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            v = lossf(model(Xva), Yva).item()
        if v < best - 1e-5:
            best, best_state, wait = v, {k: t.clone() for k, t in model.state_dict().items()}, 0
        else:
            wait += 1
            if wait >= PATIENCE:
                break
    model.load_state_dict(best_state)
    return model


def _predict(model, X):
    model.eval()
    with torch.no_grad():
        return model(torch.tensor(X, dtype=torch.float32)).numpy()


def backtest(builders, run_id=None,
             output_root=Path("artifacts/evaluation")):
    """builders: {name: fn(C) -> nn.Module}. Returns scored results DataFrame."""
    feat, _ = build(pd.read_parquet("data/ohlcv.parquet"))
    output_dir = reserve_run_dir("neural", feat.date.max(), run_id, output_root)
    win, index = channel_windows(feat)
    ytargets = [f"rv{h}" for h in range(1, H + 1)]

    recs, vol_err = [], []
    for train, test, start in run_folds(feat, DRIVER):
        train = train.dropna(subset=ytargets)
        fit, cal = split_calibration(train)

        Xfit, ofit = _lookup(win, index, fit)
        Xcal, ocal = _lookup(win, index, cal)
        Xte, ote = _lookup(win, index, test)
        Yfit = np.log(fit[ytargets].to_numpy()[ofit])
        mu, sd = Xfit.mean((0, 1)), Xfit.std((0, 1)) + 1e-8
        Xfit_s, Xcal_s, Xte_s = (_scale(x, mu, sd) for x in (Xfit, Xcal, Xte))

        cal, test = cal[ocal], test[ote]
        last = test.close.to_numpy()

        for name, make in builders.items():
            net = train_net(make(len(CHANNELS)), Xfit_s, Yfit,
                            Xcal_s, np.log(cal[ytargets].to_numpy()))
            sig_cal = clip_sigma(np.exp(_predict(net, Xcal_s)))  # (n, H)
            sig_te = clip_sigma(np.exp(_predict(net, Xte_s)))

            for h in range(1, H + 1):
                z = calibrate(cal[f"y{h}"].to_numpy(), sig_cal[:, h - 1], h)
                y = last * np.exp(test[f"y{h}"].to_numpy())
                recs.append(pd.DataFrame({
                    "model": name, "asset": test.asset.values, "origin": test.date.to_numpy(),
                    "fold": np.repeat(start, len(test)), "h": h, "y": y, "last": last,
                    "sigma": sig_te[:, h - 1], "rv": test[f"rv{h}"].to_numpy(), "regime_driver": test.vol_21d.to_numpy(),
                    **{f"q{int(q * 100)}": v
                       for q, v in bands(z, last, sig_te[:, h - 1], h).items()},
                }))
                truth = np.log(test[f"rv{h}"].to_numpy())
                vol_err.append({"model": name, "h": h,
                                "mae": np.nanmean(np.abs(truth - np.log(sig_te[:, h - 1])))})

    res = pd.concat(recs, ignore_index=True)
    save_predictions(res, output_dir, {
        "pipeline": "daily", "family": "neural", "data_end": feat.date.max(),
        "horizons": H, "folds": res.fold.nunique(),
        "origins": res[["asset", "origin"]].drop_duplicates().shape[0],
        "models": list(builders), "lookback": LOOKBACK, "channels": CHANNELS,
        "run_id": run_id, "output_dir": output_dir.as_posix(),
    }, reserved=True)
    res.attrs["output_dir"] = output_dir.as_posix()
    return res, pd.DataFrame(vol_err)


MODELS = Path("models")


def train_and_save(build_fn, name):
    """Train once on all data, save a .pt artifact, plot the split. Mirrors the
    tree train path so `python -m neural.<model>` behaves like `tree.lgbm`."""
    feat, _ = build(pd.read_parquet("data/ohlcv.parquet"))
    win, index = channel_windows(feat)
    ytargets = [f"rv{h}" for h in range(1, H + 1)]
    rows = feat.dropna(subset=ytargets + DRIVER)

    fit, cal = split_calibration(rows)
    Xfit, ofit = _lookup(win, index, fit)
    Xcal, ocal = _lookup(win, index, cal)
    mu, sd = Xfit.mean((0, 1)), Xfit.std((0, 1)) + 1e-8
    net = train_net(build_fn(len(CHANNELS)), _scale(Xfit, mu, sd),
                    np.log(fit[ytargets].to_numpy()[ofit]),
                    _scale(Xcal, mu, sd), np.log(cal[ytargets].to_numpy()[ocal]))

    # z-table from OUT-OF-SAMPLE (calibration slice) sigma, per horizon.
    calr = cal[ocal]
    sig_cal = clip_sigma(np.exp(_predict(net, _scale(Xcal, mu, sd))))
    z = {h: calibrate(calr[f"y{h}"].to_numpy(), sig_cal[:, h - 1], h) for h in range(1, H + 1)}

    art = {"model": name, "state": net.state_dict(), "channels": len(CHANNELS),
           "lookback": LOOKBACK, "mu": mu, "sd": sd, "z": z, "H": H,
           "quantiles": list(next(iter(z.values())).keys()),
           "trained_at": pd.Timestamp.now(tz="UTC"), "data_end": feat.date.max()}
    MODELS.mkdir(exist_ok=True)
    out = MODELS / f"vol7d_{name}.pt"
    torch.save(art, out)
    print(f"[{name}] trained on {len(fit)} rows, calibrated on {len(cal)}")
    print(f"saved {out} ({out.stat().st_size / 1e6:.1f} MB)")

    _plot_split(net, rows, win, index, mu, sd, name)


def _plot_split(net, rows, win, index, mu, sd, name, h=1):
    from crypto.plots import render_split, CAL_FRAC

    Xall, oall = _lookup(win, index, rows)
    pred = clip_sigma(np.exp(_predict(net, _scale(Xall, mu, sd))[:, h - 1])) * 100
    df = rows[oall].assign(pred=pred, actual=rows[oall][f"rv{h}"].to_numpy() * 100)

    panels = []
    for asset in df.asset.unique():
        g = df[df.asset == asset].sort_values("date")
        split_i = int((g.date < g.date.quantile(1 - CAL_FRAC)).sum())
        panels.append({"asset": asset, "dates": g.date.to_numpy(),
                       "actual": g.actual.to_numpy(), "pred": g.pred.to_numpy(), "split_i": split_i})
    print(f"wrote {render_split(panels, Path('plots') / f'train_split_{name}.svg')}")


def load_and_forecast(name, feat, h):
    """Load a saved neural .pt and forecast horizon h for the latest bar per coin."""
    import importlib

    art = torch.load(MODELS / f"vol7d_{name}.pt", weights_only=False)
    net = importlib.import_module(f"neural.{name}").build(art["channels"])
    net.load_state_dict(art["state"])

    win, index = channel_windows(feat, art["lookback"])
    now = feat.groupby("asset", sort=False).tail(1)
    X, ok = _lookup(win, index, now)
    now = now[ok]
    sigma = clip_sigma(np.exp(_predict(net, _scale(X, art["mu"], art["sd"]))[:, h - 1]))

    last = now.close.to_numpy()
    out = pd.DataFrame({"asset": now.asset.values, "last": last, "vol_pred_%": sigma * 100})
    for q, v in bands(art["z"][h], last, sigma, h).items():
        out[f"q{int(q * 100)}"] = v
    out["band_%"] = (out.q90 - out.q10) / out["last"] * 100
    return out.set_index("asset").sort_values("band_%"), art


def report_and_log(res, vol_err, tag):
    summary = score(res)
    n = len(res) // (res.model.nunique() * H)
    print(f"{n} origins, lookback {LOOKBACK}, channels {CHANNELS}\n")
    print("VOLATILITY MAE on log realised vol (lower better)")
    print(vol_err.pivot_table(index="h", columns="model", values="mae").round(4).to_string())
    print("\nRESULTING PRICE INTERVALS")
    print(summary.round(2).to_string())
    print("\ncoverage % by horizon day (target 80)")
    print(coverage_by_h(res).round(1).to_string())
    log(tag, summary, origins=n)
