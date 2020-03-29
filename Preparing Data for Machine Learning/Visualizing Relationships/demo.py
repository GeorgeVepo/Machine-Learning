import matplotlib.pyplot as pit
import sklearn
import pandas as pd
import numpy as np
import datetime
import seaborn as sns

# import dataset
automobile_df = pd.read_csv("../datasets/cars_processed.csv")

# ploting age x MPG
pit.figure(figsize=(6, 4))
pit.bar(automobile_df['Age'], automobile_df['MPG'])
pit.xlabel('Age')
pit.ylabel('Miles per gallon')
#pit.show()

# ploting acceleration x MPG
pit.figure(figsize=(6, 4))
pit.scatter(automobile_df['Acceleration'], automobile_df['MPG'], color='g')
pit.xlabel('Acceleration')
pit.ylabel('Miles per gallon')
#pit.show()

# ploting weight x MPG
pit.figure(figsize=(6, 4))
pit.scatter(automobile_df['Weight'], automobile_df['MPG'], color='r')
pit.xlabel('Weight')
pit.ylabel('Miles per gallon')
#pit.show()

# ploting acceleration x weight x MPG
automobile_df.plot.scatter(x='Weight',
                           y='Acceleration',
                           c='Horsepower',
                           colormap='viridis',
                           figsize=(6, 4))


# ploting Cylinders x MPG
pit.figure(figsize=(6, 4))
pit.bar(automobile_df['Cylinders'], automobile_df['MPG'])
pit.xlabel('Cylinders')
pit.ylabel('Miles per gallon')
#pit.show()

# drop columns
automobile_df.drop(['Cylinders', 'Origin'], axis=1, inplace=True)

cars_corr = automobile_df.corr()
fig, ax = pit.subplots(figsize=(6, 4))
sns.heatmap(cars_corr, annot=True)
cars_corr