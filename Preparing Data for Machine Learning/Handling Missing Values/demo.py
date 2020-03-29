import sklearn
import pandas as pd
import numpy as np
import datetime

# import dataset
automobile_dfe = pd.read_csv("../datasets/cars.csv")

# Replace ? values to nan
automobile_dfe = automobile_dfe.replace("?", np.nan)

# Replace nan to mean
automobile_dfe["MPG"] = automobile_dfe["MPG"].fillna(automobile_dfe["MPG"].mean())

# Drop nan values
automobile_dfe = automobile_dfe.dropna()

# Drop features
automobile_dfe.drop(["Model", "compression-ratio", "bore", "stroke"], axis=1, inplace=True)

# Verifying how many numeric values contains in the year column
print(automobile_dfe['Year'].str.isnumeric().value_counts())

# Cleaning year column
extr = automobile_dfe['Year'].str.extract(r'^(\d{4})', expand=False)
automobile_dfe['Year'] = pd.to_numeric(extr)

# Replace year by age
automobile_dfe['Age'] = datetime.datetime.now().year - automobile_dfe['Year'];
automobile_dfe.drop(["Year"], axis=1, inplace=True)

# Replace Cylinders "-" values to mean
cylinders = automobile_dfe['Cylinders'].loc[automobile_dfe['Cylinders'] != '-']
cmean = cylinders.astype(int).mean()
automobile_dfe['Cylinders'] = automobile_dfe['Cylinders'].replace('-', cmean).astype(int)

# Parse columns values to numeric or nan
automobile_dfe['Displacement'] = pd.to_numeric(automobile_dfe['Displacement'], errors='coerce')
automobile_dfe['Weight'] = pd.to_numeric(automobile_dfe['Weight'], errors='coerce')
automobile_dfe['Acceleration'] = pd.to_numeric(automobile_dfe['Acceleration'], errors='coerce')


# Cleaning origin column
print(automobile_dfe['Origin'].unique())

automobile_dfe['Origin'] = np.where(automobile_dfe['Origin'].str.contains('US'),
                                    'US',
                                    automobile_dfe['Origin'])

automobile_dfe['Origin'] = np.where(automobile_dfe['Origin'].str.contains('Japan'),
                                    'Japan',
                                    automobile_dfe['Origin'])

automobile_dfe['Origin'] = np.where(automobile_dfe['Origin'].str.contains('Europe'),
                                    'Europe',
                                    automobile_dfe['Origin'])

print(automobile_dfe.dtypes())
print(automobile_dfe.head())
automobile_dfe.to_csv('../datasets/cars_processed.csv', index=False)
