import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import math
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

# Display floating point values and numpy arrays rounded of to three places after the decimal
np.set_printoptions(precision=3)

diabetes_df = pd.read_csv('../datasets/diabetes_processed.csv')
diabetes_df.head(10)

feature_df = diabetes_df.drop('Outcome', axis=1)
target_df = diabetes_df['Outcome']

print(feature_df.shape)
print(feature_df.describe())

# Min and max between 0 and 1
scaler = MinMaxScaler(feature_range=(0, 1))
rescaled_features = scaler.fit_transform(feature_df)
rescaled_features = pd.DataFrame(rescaled_features, columns=feature_df.columns)
print(rescaled_features.head(5))

print(rescaled_features.describe()[3:4])
print(rescaled_features.describe()[7:8])

# Calculate the mean then subtract this mean or average value from each feature values
# and divide by the standard deviation. It make the data be around of 0
scaler = StandardScaler()
scaler.fit(feature_df)
standardized_features = scaler.transform(feature_df)
standardized_features_df = pd.DataFrame(standardized_features, columns=feature_df.columns)
print(rescaled_features.describe()[1:2], rescaled_features.describe()[2:3])