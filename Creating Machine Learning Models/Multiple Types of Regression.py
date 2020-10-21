import ml_utils
import pandas as pd

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

data = pd.read_csv('datasets/diamonds_processed.csv', index_col=0)

ml_utils.linear_regression(data, "price")
ml_utils.lasso_linear_regression(data, "price")
ml_utils.ridge_linear_regression(data, "price")
ml_utils.sgd_linear_regression(data, "price")