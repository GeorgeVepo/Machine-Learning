import pandas as pd
import numpy as np
from sklearn.linear_model import Lasso

import ml_utils
from ml_utils import TransformMethod, FeatureSelectionMethod

automobile_df = pd.read_csv('../datasets/cars_processed.csv')


automobile_df = automobile_df.drop('Origin', axis=1)

ml_utils.linear_regression(automobile_df, 'MPG',
                             transform_method=TransformMethod.standard_scaler,
                             feature_selection_method=FeatureSelectionMethod.lasso)

ml_utils.linear_regression(automobile_df, 'MPG',
                             transform_method=TransformMethod.standard_scaler,
                             feature_selection_method=FeatureSelectionMethod.decision_tree)
