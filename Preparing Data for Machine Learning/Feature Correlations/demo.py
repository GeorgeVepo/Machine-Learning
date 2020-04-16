import pandas as pd
import matplotlib.pyplot as plt
import seaborn as ans

from ml_utils import config_print
from yellowbrick.target import FeatureCorrelation

config_print()

diabetes = pd.read_csv('../datasets/diabetes.csv')
print(diabetes.corr())

plt.figure(figsize=(12, 8))
ans.heatmap(diabetes.corr(), annot=True)
plt.show()

x = diabetes[['Insulin', 'BMI', 'BloodPressure', 'DiabetesPedigreeFunction']]
y = diabetes['Age']

feature_names = x.columns
visualizer = FeatureCorrelation(labels=feature_names, method='pearson')
visualizer.fit(x, y)

visualizer.show()

score_df = pd.DataFrame({'Feature_names' : visualizer.features_, 'Scores' : visualizer.scores_})
print(score_df)

x = diabetes.drop('Outcome', axis=1)
y = diabetes['Outcome']

visualizer = FeatureCorrelation(labels=x.columns, method='pearson')
visualizer.fit(x, y)

visualizer.show()

# mutual_info Need to declare what features are discrete (Pregnancy)
discrete_features = [False for _ in range(len(x.columns))]
discrete_features[0] = True

visualizer = FeatureCorrelation(method='mutual_info-classification',
                                labels=x.columns, sort=True)

visualizer.fit(x, y, discrete_features=discrete_features, random_state=0)

visualizer.show()

