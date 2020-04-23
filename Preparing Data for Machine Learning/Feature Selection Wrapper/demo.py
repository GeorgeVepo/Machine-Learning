import pandas as pd
import numpy as np

import ml_utils
from ml_utils import TransformMethod, FeatureSelectionMethod

diabetes_data = pd.read_csv('../datasets/diabetes.csv')

ml_utils.logistic_regression(diabetes_data, 'Outcome',
                             transform_method=TransformMethod.standard_scaler,
                             feature_selection_method=FeatureSelectionMethod.rfe)

ml_utils.logistic_regression(diabetes_data, 'Outcome',
                             transform_method=TransformMethod.standard_scaler,
                             feature_selection_method=FeatureSelectionMethod.forward)

ml_utils.logistic_regression(diabetes_data, 'Outcome',
                             transform_method=TransformMethod.standard_scaler,
                             feature_selection_method=FeatureSelectionMethod.backward)
