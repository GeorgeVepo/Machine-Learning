import pandas as pd
import numpy as np

from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

diabetes = pd.read_csv('../datasets/diabetes_processed.csv')
diabetes_features = diabetes.drop('Outcome', axis=1)
diabetes_label = diabetes[['Outcome']]

# Adding randomly null values on dataset
mask = np.random.randint(0, 100, size=diabetes_features.shape).astype(np.bool)
mask = np.logical_not(mask)
diabetes_features[mask] = np.nan
print(diabetes_features.isnull().sum())

# Running classification
x_train, x_test, y_train, y_test = train_test_split(diabetes_features, diabetes_label, test_size=0.2)
transformer = ColumnTransformer(transformers=[('features', SimpleImputer(strategy='mean'), [0, 1, 2, 3, 4, 5, 6, 7])])
clf = make_pipeline(transformer, DecisionTreeClassifier(max_depth=4))
clf = clf.fit(x_train, y_train)
print(clf.score(x_train, y_train))
y_pred = clf.predict(x_test)
print(accuracy_score(y_pred, y_test))