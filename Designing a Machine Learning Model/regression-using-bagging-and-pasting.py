import pandas as pd
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import BaggingRegressor

insurance_data = pd.read_csv('dataset/insurance_processed.csv')

x = insurance_data.drop('charges', axis=1)
y = insurance_data['charges']

x_train, x_test, y_train, y_test =   train_test_split(x, y, test_size=0.2)

#####################Bagging####################
bag_reg = BaggingRegressor(DecisionTreeRegressor(),
                           n_estimators=500,
                           bootstrap=True,
                           max_samples=0.8,
                           n_jobs=1,
                           oob_score=True)

bag_reg.fit(x_train, y_train)

print(bag_reg.oob_score_)

y_pred = bag_reg.predict(x_test)
print(r2_score(y_test, y_pred))

#####################Pasting####################
bag_reg = BaggingRegressor(DecisionTreeRegressor(),
                           n_estimators=500,
                           bootstrap=False,
                           max_samples=0.9,
                           n_jobs=1)

bag_reg.fit(x_train, y_train)

y_pred = bag_reg.predict(x_test)
print(r2_score(y_test, y_pred))