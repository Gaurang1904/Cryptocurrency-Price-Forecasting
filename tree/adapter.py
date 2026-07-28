"""Tree adapter.

Trees split on raw values, tolerate any scale, and handle NaN natively. So the
adapter is the identity - this file exists to say that out loud. The whole
reason gradient boosting is low-friction on tabular data lives in these two lines.
"""

def inputs(feat, cols):
    return feat[cols]  # ponytail: trees need no scaling, no imputation, nothing
