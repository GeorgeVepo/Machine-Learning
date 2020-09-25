import sklearn
import pandas as pd
import numpy as np
import statsmodels.api as sm

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

advertising_data = pd.read_csv('dataset/Advertising.csv', index_col=0)

X = advertising_data.drop('sales', axis=1)
Y = advertising_data['sales']

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.30, random_state=0)


# perform analytical technique
x_train_with_intercept = sm.add_constant(x_train)
stats_model = sm.OLS(y_train, x_train_with_intercept)

fit_model = stats_model.fit()
print(fit_model.summary())





