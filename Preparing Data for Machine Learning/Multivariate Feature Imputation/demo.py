import pandas as pd
import numpy as np
from sklearn.impute._iterative import IterativeImputer


diabetes = pd.read_csv('../datasets/diabetes_processed_incomplete.csv')

# Performing multivariate feature imputation
diabetes_feature = diabetes.drop('Outcome', axis=1)
print(diabetes_feature['Insulin'].head(10))
diabetes_label = diabetes[['Outcome']]
imp = IterativeImputer(max_iter=100, random_state=0)
imp.fit(diabetes_feature)
diabetes_feature_arr = imp.transform(diabetes_feature)

# Getting data together again
diabetes_feature = pd.DataFrame(diabetes_feature_arr, columns=diabetes_feature.columns)
print(diabetes_feature['Insulin'].head(10))
print(diabetes_feature.isnull().sum())

