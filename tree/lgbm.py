"""LightGBM volatility regressor. Target = log realised vol.

Train + save + plot:  python -m tree.lgbm
This is the deployed model - it saves to models/vol7d.joblib, which predict.py loads.
"""

import numpy as np
import lightgbm as lgb

PARAMS = dict(n_estimators=300, learning_rate=0.05, num_leaves=31,
              min_child_samples=50, subsample=0.8, subsample_freq=1,
              colsample_bytree=0.8, verbose=-1)


def fit_vol(fit, cols, h, params=None):
    from tree.adapter import inputs
    return lgb.LGBMRegressor(**(params or PARAMS)).fit(inputs(fit, cols), np.log(fit[f"rv{h}"]))


if __name__ == "__main__":
    from crypto.train import train_and_save
    train_and_save(fit_vol, "lgbm")
