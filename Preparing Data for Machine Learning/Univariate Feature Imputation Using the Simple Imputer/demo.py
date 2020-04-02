import pandas as pd
import numpy as np

# Provides basic strategies for imputing missing values(constant, mean, median, mode)
from sklearn.impute import SimpleImputer


# import dataset
diabetes = pd.read_csv("../datasets/diabetes.csv")
diabetes.info()

print(diabetes.describe().transpose()['min'])

diabetes['Glucose'].replace(0, np.nan, inplace=True)
diabetes['BloodPressure'].replace(0, np.nan, inplace=True)
diabetes['SkinThickness'].replace(0, np.nan, inplace=True)
diabetes['Insulin'].replace(0, np.nan, inplace=True)
diabetes['BMI'].replace(0, np.nan, inplace=True)

imp = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
imp.fit(diabetes['SkinThickness'].values.reshape(-1, 1))
diabetes['SkinThickness'] = imp.transform(diabetes['SkinThickness'].values.reshape(-1, 1))

imp = SimpleImputer(missing_values=np.nan, strategy='median')
imp.fit(diabetes['Glucose'].values.reshape(-1, 1))
diabetes['Glucose'] = imp.transform(diabetes['Glucose'].values.reshape(-1, 1))

imp = SimpleImputer(missing_values=np.nan, strategy='mean')
imp.fit(diabetes['BloodPressure'].values.reshape(-1, 1))
diabetes['BloodPressure'] = imp.transform(diabetes['BloodPressure'].values.reshape(-1, 1))

# Just fill the missing values with 32 that is the mean values of BMI
imp = SimpleImputer(missing_values=np.nan, strategy='constant', fill_value=32)
imp.fit(diabetes['BMI'].values.reshape(-1, 1))
diabetes['BMI'] = imp.transform(diabetes['BMI'].values.reshape(-1, 1))

