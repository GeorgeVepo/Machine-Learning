import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import ml_utils as ml

automobile_df = pd.read_csv('../datasets/cars_processed.csv')
automobile_df.head()

# Plotting Mpg and age to see the relationship
x = automobile_df[['Age']]
y = automobile_df['MPG']
fig, ax = plt.subplots(figsize=(6,4))
plt.scatter(automobile_df['Age'], automobile_df['MPG'])
plt.xlabel('Age')
plt.ylabel('Mpg')
#plt.show()

# Linear Regression, age to predict MPG
# Test size of 20%, test split will shuffle our data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
linear_model = LinearRegression(normalize=True).fit(x_train, y_train)

# The score using the training data

# The score using the test data
y_pred = linear_model.predict(x_test)

# Plotting our test data vs linear regression prediction line
fig, ax = plt.subplots(figsize=(6,4))
plt.scatter(x_test, y_test)
plt.plot(x_test, y_pred, color='r')
plt.xlabel('Age')
plt.ylabel('Mpg')
#plt.show()

# Performing Liner Regression using Horsepower vs MPG
x = automobile_df[['Horsepower']]
y = automobile_df['MPG']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
linear_model = LinearRegression(normalize=True).fit(x_train, y_train)
y_pred = linear_model.predict(x_test)
fig, ax = plt.subplots(figsize=(6,4))
plt.scatter(x_test, y_test)
plt.plot(x_test, y_pred, color='r')
plt.xlabel('Horsepower')
plt.ylabel('Mpg')
#plt.show()

# Converting origin to numeric
automobile_df = pd.get_dummies(automobile_df, columns=['Origin'])

# Performing multi variable Liner Regression
ml.linear_regression(automobile_df, 'MPG')

# Performing one variable Liner Regression with discretization
ml.univariable_linear_regression(automobile_df, 'Horsepower', 'MPG', n_bins=20, discretization=True)




