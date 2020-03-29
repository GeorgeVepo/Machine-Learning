import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

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
print("Training Scrore", linear_model.score(x_train, y_train))

# The score using the test data
y_pred = linear_model.predict(x_test)
print('Testing score', r2_score(y_test, y_pred))

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
print("Training Scrore", linear_model.score(x_train, y_train))
y_pred = linear_model.predict(x_test)
print('Testing score', r2_score(y_test, y_pred))
fig, ax = plt.subplots(figsize=(6,4))
plt.scatter(x_test, y_test)
plt.plot(x_test, y_pred, color='r')
plt.xlabel('Horsepower')
plt.ylabel('Mpg')
#plt.show()

# Converting origin to numeric
automobile_df = pd.get_dummies(automobile_df, columns=['Origin'])
print(automobile_df.head())

# Performing multi variable Liner Regression
x = automobile_df.drop('MPG', axis=1)
y = automobile_df['MPG']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
linear_model = LinearRegression(normalize=True).fit(x_train, y_train)
print("Training Scrore", linear_model.score(x_train, y_train))
y_pred = linear_model.predict(x_test)
print('Testing score', r2_score(y_test, y_pred))


# Predicting MPG from a data row
row = automobile_df.head(1)
x = row.drop('MPG', axis=1)
y = row['MPG']
y_pred = linear_model.predict(x)
print('Data to predict', y)
print('Predicted data', y_pred)


