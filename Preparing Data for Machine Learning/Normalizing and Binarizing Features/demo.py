from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import Binarizer
import ml_utils as ml
import pandas as pd
import numpy as np

diabetes_df = pd.read_csv('../datasets/diabetes_processed.csv')

# l1 magnitude, sum of the absolute values of a row is equal to 1
x, y = ml.transform_data(diabetes_df, 'Outcome', ml.TransformMethod.normalize_l1)
print(x.iloc[0])
print(x.iloc[0].abs().sum())


# l2 magnitude, sum of the squares of the values of a row is equal to 1
x, y = ml.transform_data(diabetes_df, 'Outcome', ml.TransformMethod.normalize_l2)
print(x.iloc[0])
print(x.iloc[0].pow(2).sum())

# max magnitude, the highest value in a feature will be equal to 1 and the
# other values of that feature will be expressed in terms of this max
x, y = ml.transform_data(diabetes_df, 'Outcome', ml.TransformMethod.normalize_max)
print(x.head())

# Will transform the column values of all features in boolean values
x, y = ml.transform_data(diabetes_df, 'Outcome', ml.TransformMethod.binarize)
print(x.head())

ml.logistic_regression(diabetes_df, 'Outcome', ml.TransformMethod.min_max_scaler)
ml.logistic_regression(diabetes_df, 'Outcome', ml.TransformMethod.standard_scaler)
ml.logistic_regression(diabetes_df, 'Outcome', ml.TransformMethod.normalize_l1)
ml.logistic_regression(diabetes_df, 'Outcome', ml.TransformMethod.normalize_l2)
ml.logistic_regression(diabetes_df, 'Outcome', ml.TransformMethod.normalize_max)
ml.logistic_regression(diabetes_df, 'Outcome', ml.TransformMethod.binarize)


