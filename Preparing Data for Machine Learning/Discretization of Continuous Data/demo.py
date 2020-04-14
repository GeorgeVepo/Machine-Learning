import pandas as pd
import numpy as np
from sklearn.preprocessing import KBinsDiscretizer
from ml_utils import config_print

config_print()

x = np.array([-7, 2, -3, -11, 14, 6, 8])

categories = pd.cut(x, 4)
print("1 - ", categories)
print("2 - ", categories.categories)
print("3 - ", categories.codes)

categories = pd.cut(x, 4, retbins=True)
print("4 - ", categories)

marks = np.array([70, 20, 30, 99, 40, 16, 80])
categories, bins = pd.cut(marks, 4, retbins=True, labels=['poor', 'average', 'good', 'excellent'])
print("5 - ", categories)

marks = [[70], [20], [30], [99], [40], [16], [80]]
# Ordinal (numeric dividers), uniform (all groups will have the same limit proportions)
enc = KBinsDiscretizer(n_bins=4, encode='ordinal', strategy='uniform')
enc.fit(marks)
marks_trans = enc.transform(marks)
print("6 - ", marks_trans)
print("7 - ", enc.bin_edges_)

x = [[-21, 41, -14],
     [-13, 23, -31],
     [9, 30, -5],
     [0, 24, -17]]

# quantile (every group will have the same number of values)
enc = KBinsDiscretizer(n_bins=4, encode='ordinal', strategy='quantile')
enc.fit(x)
x_trans = enc.transform(x)
print("8 - ", x_trans)
print("9 - ", enc.bin_edges_)
# will try to recreate the original array based on the edges points, by their average values
x_trans = enc.inverse_transform(x_trans)
print("10 - ", x_trans)