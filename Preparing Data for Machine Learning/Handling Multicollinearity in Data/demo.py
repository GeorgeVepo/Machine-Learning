import pandas as pd
from sklearn import preprocessing
from statsmodels.stats.outliers_influence import variance_inflation_factor

import ml_utils

automobile_df = pd.read_csv('../datasets/cars_processed.csv')

# Standardize all features
automobile_df[['Cylinders']] = preprocessing.scale(automobile_df[['Cylinders']].astype('float64'))
automobile_df[['Displacement']] = preprocessing.scale(automobile_df[['Displacement']].astype('float64'))
automobile_df[['Horsepower']] = preprocessing.scale(automobile_df[['Horsepower']].astype('float64'))
automobile_df[['Weight']] = preprocessing.scale(automobile_df[['Weight']].astype('float64'))
automobile_df[['Acceleration']] = preprocessing.scale(automobile_df[['Acceleration']].astype('float64'))
automobile_df[['Age']] = preprocessing.scale(automobile_df[['Age']].astype('float64'))


x = automobile_df.drop('Origin', axis=1)
ml_utils.linear_regression(x, 'MPG')

# Finding multicollinearity by correlation metrics
x = automobile_df.drop('MPG', axis=1)
print(abs(x.corr()) > 0.8)
x = automobile_df.drop(['Origin', 'Cylinders', 'Displacement', 'Weight'], axis=1)
ml_utils.linear_regression(x, 'MPG')

# Finding multicollinearity by variance_inflation_factor
x = automobile_df.drop(['Origin', 'MPG'], axis=1)
vif = ml_utils.variance_inflation_factor_calculator(x)
x = automobile_df.drop(['Origin', 'Displacement', 'Weight'], axis=1)
vif = ml_utils.variance_inflation_factor_calculator(x)
ml_utils.linear_regression(x, 'MPG')

