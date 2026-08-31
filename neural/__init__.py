"""Neural family: LSTM / N-HiTS / TFT, including the next-day TFT pipeline.

Adapter will window features into 3D sequences and scale to ~[-1,1] - the two
things neural nets need that trees and linear models do not. Requires
neuralforecast or torch; run `pip install neuralforecast` before building.

The TFT implementation is complete for user-operated 15-minute next-day
experiments; full model fitting is intentionally not run automatically.
"""
