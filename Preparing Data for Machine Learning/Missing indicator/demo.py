import pandas as pd
import numpy as np

from sklearn.impute import MissingIndicator

features = [[4, 2, 1],
            [24, 12, 6],
            [8, 4, 2],
            [28, 14, 7],
            [32, 16, -1],
            [600, 300, 150],
            [-1, 60, 30],
            [-1, 4, 1]]


indicator = MissingIndicator(missing_values=-1)
mask_missing_values_only = indicator.fit_transform(features)
print(mask_missing_values_only)
print("Read features (only features with missing values)", indicator.features_)

indicator = MissingIndicator(missing_values=-1, features="all")
mask_missing_values_only = indicator.fit_transform(features)
print(mask_missing_values_only)
print("Read features (all)", indicator.features_)
