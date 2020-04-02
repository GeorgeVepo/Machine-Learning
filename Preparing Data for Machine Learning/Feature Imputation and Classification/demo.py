import pandas as pd
import numpy as np

from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

diabetes = pd.read_csv('../datasets/diabetes_processed.csv')

diabetes_features = diabetes.drop('Outcome', axis=1)
diabetes_label = diabetes[['Outcome']]

mask = np.random.randint(0, 100, size=diabetes_features.shape).astype(np.bool)
mask = np.logical_not(mask)

diabetes_features[mask] = np.nan
print(diabetes_features.isnull().sum())
print()

