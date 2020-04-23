import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score

from ml_utils import logistic_regression_model
from ml_utils import FeatureSelectionMethod
from ml_utils import TransformMethod
from sklearn.feature_selection import chi2
from sklearn.feature_selection import SelectKBest

x = pd.read_csv('../datasets/diabetes.csv')

diabetes_data = x
diabetes_data['Glucose'].replace(0, np.nan, inplace=True)
diabetes_data['BloodPressure'].replace(0, np.nan, inplace=True)
diabetes_data['SkinThickness'].replace(0, np.nan, inplace=True)
diabetes_data['Insulin'].replace(0, np.nan, inplace=True)
diabetes_data['BMI'].replace(0, np.nan, inplace=True)

print(diabetes_data.isnull().sum())

# Percentage of the data is null
print(diabetes_data['Glucose'].isnull().sum() / len(diabetes_data) * 100)
print(diabetes_data['BloodPressure'].isnull().sum() / len(diabetes_data) * 100)
print(diabetes_data['SkinThickness'].isnull().sum() / len(diabetes_data) * 100)
print(diabetes_data['Insulin'].isnull().sum() / len(diabetes_data) * 100)

# Will drop the column that have more than 10 percent of its data as null
diabetes_data_trim = diabetes_data.dropna(thresh=int(diabetes_data.shape[0] * .9), axis=1)
print(diabetes_data_trim.columns)


model, x_test, y_test = logistic_regression_model(x, 'Outcome',
                                                  transform_method=TransformMethod.standard_scaler,
                                                  feature_selection_method=FeatureSelectionMethod.none)

y_pred = model.predict(x_test)

print("Testing score none : ", accuracy_score(y_test, y_pred))


model, x_test, y_test = logistic_regression_model(x, 'Outcome',
                                                  transform_method=TransformMethod.standard_scaler,
                                                  feature_selection_method=FeatureSelectionMethod.missing_value_ratio)
y_pred = model.predict(x_test)
print("Testing score missing_value_ratio : ", accuracy_score(y_test, y_pred))


model, x_test, y_test = logistic_regression_model(x, 'Outcome',
                                                  transform_method=TransformMethod.standard_scaler,
                                                  feature_selection_method=FeatureSelectionMethod.chi2)
y_pred = model.predict(x_test)
print("Testing score chi2 : ", accuracy_score(y_test, y_pred))


model, x_test, y_test = logistic_regression_model(x, 'Outcome',
                                                  transform_method=TransformMethod.standard_scaler,
                                                  feature_selection_method=FeatureSelectionMethod.anova)

y_pred = model.predict(x_test)
print("Testing score anova : ", accuracy_score(y_test, y_pred))

