"""Train/test split plots as hand-rolled SVG - no matplotlib.

matplotlib's native _image DLL is blocked by Windows Application Control on this
machine, and even importing pyplot trips it. The plot here is just two polylines
plus a split marker per coin, so raw SVG covers it with zero dependencies and no
blockable binaries. Output renders in VS Code, browsers, and the file panel.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from crypto.features import H
from crypto.model import CAL_FRAC, clip_sigma

PLOTS = Path("plots")
W, PH, PAD = 1000, 190, 55  # panel width, height, margin


def _poly(xs, ys, y0, y1, x0, x1, top, h):
    """Map data to SVG panel coords -> 'x,y x,y ...' for a polyline."""
    sx = (W - 2 * PAD) / max(x1 - x0, 1)
    sy = (h - 2 * 12) / max(y1 - y0, 1e-9)
    pts = [(PAD + (x - x0) * sx, top + h - 12 - (y - y0) * sy) for x, y in zip(xs, ys)]
    return " ".join(f"{px:.1f},{py:.1f}" for px, py in pts)


def _panel(top, title, dates, actual, pred, split_i):
    n = len(dates)
    ymax = float(np.nanmax(actual)) or 1.0
    x0, x1 = 0, n - 1
    sx = (W - 2 * PAD) / max(x1, 1)
    split_x = PAD + split_i * sx

    # year gridlines
    years = pd.DatetimeIndex(dates).year
    ticks = "".join(
        f'<line x1="{PAD + i * sx:.1f}" y1="{top}" x2="{PAD + i * sx:.1f}" y2="{top + PH}" '
        f'stroke="#eee"/><text x="{PAD + i * sx:.1f}" y="{top + PH + 12}" font-size="9" '
        f'fill="#999" text-anchor="middle">{years[i]}</text>'
        for i in range(1, n) if years[i] != years[i - 1])

    return f'''<g>
<rect x="{PAD}" y="{top + PH - 12 - (PH - 24)}" width="{W - 2 * PAD}" height="{PH - 24}" fill="none"/>
<rect x="{split_x:.1f}" y="{top}" width="{W - PAD - split_x:.1f}" height="{PH}" fill="#1f77b4" opacity="0.06"/>
{ticks}
<polyline points="{_poly(range(n), actual, 0, ymax, x0, x1, top, PH)}" fill="none" stroke="#444" stroke-width="0.7"/>
<polyline points="{_poly(range(n), pred, 0, ymax, x0, x1, top, PH)}" fill="none" stroke="#e45756" stroke-width="0.9"/>
<line x1="{split_x:.1f}" y1="{top}" x2="{split_x:.1f}" y2="{top + PH}" stroke="#1f77b4" stroke-dasharray="4" stroke-width="1"/>
<text x="{PAD}" y="{top + 12}" font-size="11" fill="#333">{title}  (left = train, shaded = held-out test)</text>
<text x="{PAD - 6}" y="{top + 14}" font-size="8" fill="#999" text-anchor="end">{ymax:.0f}%</text>
</g>'''


def render_split(panels, out):
    """panels: list of dict(asset, dates, actual, pred, split_i)."""
    height = PAD + len(panels) * (PH + 24)
    body = [
        '<text x="{}" y="20" font-size="12" fill="#333">daily vol %  '
        '<tspan fill="#444">actual</tspan> / <tspan fill="#e45756">predicted</tspan></text>'.format(PAD)
    ]
    for i, p in enumerate(panels):
        body.append(_panel(35 + i * (PH + 24), p["asset"], p["dates"],
                           p["actual"], p["pred"], p["split_i"]))
    svg = (f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{height}" '
           f'font-family="sans-serif">\n' + "\n".join(body) + "\n</svg>")
    PLOTS.mkdir(exist_ok=True)
    out.write_text(svg, encoding="utf-8")
    return out


def render_forecast(panels, out, title="actual vs forecast"):
    """Fan chart per coin: actual line, median line, shaded q10-q90 band.

    panels: list of dict(asset, dates, actual, q10, q50, q90).
    """
    height = 40 + len(panels) * (PH + 24)
    body = [f'<text x="{PAD}" y="20" font-size="12" fill="#333">{title}  '
            f'<tspan fill="#444">actual</tspan> / <tspan fill="#e45756">median</tspan> / '
            f'<tspan fill="#e45756" opacity="0.5">80% band</tspan></text>']
    for i, p in enumerate(panels):
        top = 35 + i * (PH + 24)
        n = len(p["dates"])
        ys = np.concatenate([p["q10"], p["q90"], p["actual"]])
        y0, y1 = float(np.nanmin(ys)), float(np.nanmax(ys))
        band_up = _poly(range(n), p["q90"], y0, y1, 0, n - 1, top, PH).split()
        band_dn = _poly(range(n), p["q10"], y0, y1, 0, n - 1, top, PH).split()[::-1]
        body.append(f'''<g>
<polygon points="{" ".join(band_up + band_dn)}" fill="#e45756" opacity="0.15"/>
<polyline points="{_poly(range(n), p["actual"], y0, y1, 0, n - 1, top, PH)}" fill="none" stroke="#444" stroke-width="1"/>
<polyline points="{_poly(range(n), p["q50"], y0, y1, 0, n - 1, top, PH)}" fill="none" stroke="#e45756" stroke-width="1"/>
<text x="{PAD}" y="{top + 12}" font-size="11" fill="#333">{p["asset"]}</text>
</g>''')
    svg = (f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{height}" '
           f'font-family="sans-serif">\n' + "\n".join(body) + "\n</svg>")
    PLOTS.mkdir(exist_ok=True)
    out.write_text(svg, encoding="utf-8")
    return out


def render_history_forecast(panels, out, title="history + forecast"):
    """Recent price history (blue) + forecast median (orange) + shaded 80% band.

    panels: list of dict(asset, hist, q10, q50, q90). `hist` is recent prices;
    the forecast arrays start where history ends (connected).
    """
    height = 40 + len(panels) * (PH + 24)
    body = [f'<text x="{PAD}" y="20" font-size="12" fill="#333">{title}  '
            f'<tspan fill="#1f77b4">historical</tspan> / <tspan fill="#e45756">forecast</tspan> / '
            f'<tspan fill="#e45756" opacity="0.4">80% band</tspan></text>']
    for i, p in enumerate(panels):
        top = 35 + i * (PH + 24)
        nh, nf = len(p["hist"]), len(p["q50"])
        n = nh + nf
        allv = np.concatenate([p["hist"], p["q10"], p["q90"]])
        y0, y1 = float(np.nanmin(allv)), float(np.nanmax(allv))
        # forecast x starts at the last history point (connected)
        fc_x = range(nh - 1, nh - 1 + nf + 1)
        join = np.concatenate([[p["hist"][-1]], p["q50"]])
        j10 = np.concatenate([[p["hist"][-1]], p["q10"]])
        j90 = np.concatenate([[p["hist"][-1]], p["q90"]])
        band = (_poly(fc_x, j90, y0, y1, 0, n - 1, top, PH).split()
                + _poly(fc_x, j10, y0, y1, 0, n - 1, top, PH).split()[::-1])
        body.append(f'''<g>
<polygon points="{" ".join(band)}" fill="#e45756" opacity="0.15"/>
<polyline points="{_poly(range(nh), p["hist"], y0, y1, 0, n - 1, top, PH)}" fill="none" stroke="#1f77b4" stroke-width="1.1"/>
<polyline points="{_poly(fc_x, join, y0, y1, 0, n - 1, top, PH)}" fill="none" stroke="#e45756" stroke-width="1.4"/>
<text x="{PAD}" y="{top + 12}" font-size="11" fill="#333">{p["asset"]}</text>
</g>''')
    svg = (f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{height}" '
           f'font-family="sans-serif">\n' + "\n".join(body) + "\n</svg>")
    PLOTS.mkdir(exist_ok=True)
    out.write_text(svg, encoding="utf-8")
    return out


def train_split(feat, cols, models, h=1, out=None):
    """Tree/linear models: per-coin actual vs predicted vol, train/test split."""
    out = out or PLOTS / "train_split.svg"
    panels = []
    for asset in feat.asset.unique():
        g = feat[feat.asset == asset].dropna(subset=cols + [f"rv{h}"]).sort_values("date")
        if g.empty:
            continue
        split_i = int((g.date < g.date.quantile(1 - CAL_FRAC)).sum())
        pred = clip_sigma(np.exp(models[h].predict(g[cols]))) * 100
        panels.append({"asset": asset, "dates": g.date.to_numpy(),
                       "actual": g[f"rv{h}"].to_numpy() * 100, "pred": pred, "split_i": split_i})
    return render_split(panels, out)
