import pandas as pd
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import BaggingRegressor, GradientBoostingRegressor

insurance_data = pd.read_csv('dataset/insurance_processed.csv')

x = insurance_data.drop('charges', axis=1)
y = insurance_data['charges']

x_train, x_test, y_train, y_test =   train_test_split(x, y, test_size=0.2)

#####################Boosting####################
gbr = GradientBoostingRegressor(max_depth=3, n_estimators=3, learning_rate=1.0)

gbr.fit(x_train, y_train)

y_pred = gbr.predict(x_test)

print(r2_score(y_test, y_pred))
