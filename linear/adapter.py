"""Linear adapter.

Unlike trees, linear models need bounded, scaled inputs: one outlier row pushed
exp(prediction) past 1e120 without this. The three linear models take different
inputs, so the adapter is a builder, not a single transform:
  ridge -> all feature cols, RobustScaler + clip
  har   -> 3 raw realised-vol columns, no scaling needed (already log-vol)
  garch -> the return series only, no features at all
"""

import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, RobustScaler
from sklearn.linear_model import Ridge

HAR_COLS = ["har_1d", "har_5d", "har_22d"]


def ridge_pipeline(alpha=1.0):
    return make_pipeline(
        RobustScaler(),
        FunctionTransformer(lambda x: np.clip(np.nan_to_num(x), -10, 10)),
        Ridge(alpha=alpha),
    )
