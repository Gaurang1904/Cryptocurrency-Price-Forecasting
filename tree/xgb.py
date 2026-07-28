"""XGBoost volatility regressor. Same target and adapter as LightGBM.

Params mirror tree/lgbm.py so the comparison is model-vs-model, not
tuning-vs-tuning: same depth, learning rate, subsampling, tree count.

Train + save + plot:  python -m tree.xgb
"""

import numpy as np
from xgboost import XGBRegressor

PARAMS = dict(n_estimators=300, learning_rate=0.05, max_depth=5,
              min_child_weight=50, subsample=0.8, colsample_bytree=0.8,
              tree_method="hist", verbosity=0)


def fit_vol(fit, cols, h, params=None):
    from tree.adapter import inputs
    return XGBRegressor(**(params or PARAMS)).fit(inputs(fit, cols), np.log(fit[f"rv{h}"]))


if __name__ == "__main__":
    from crypto.train import train_and_save
    train_and_save(fit_vol, "xgb")
