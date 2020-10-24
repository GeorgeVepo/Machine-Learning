import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

import ml_utils as utils

data = pd.read_csv('datasets/german_credit_data_dataset.csv')

data = data.drop(['telephone', 'personal', 'present_residence', 'other_installment_plans'], axis=1)

# Categorical column with value order
savings_dict = {"A65": 0, "A61": 1, "A62": 2, "A63": 3, "A64": 4}
data['savings'].replace(savings_dict, inplace=True)

# Categorical columns with no order
data = pd.get_dummies(data, columns=['checking_account_status',
                                     'credit_history',
                                     'purpose',
                                     'present_employment',
                                     'property',
                                     'housing',
                                     'other_debtors',
                                     'job',
                                     'foreign_worker'])

x, y = utils.set_apart(data, "customer_type")
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
x_train_1, x_train_2, y_train_1, y_train_2 = train_test_split(x_train, y_train, test_size=0.5)

#Max_depth of each decision tree
#n_estimators = number of decision trees
rfc = RandomForestClassifier(max_depth=4, n_estimators=2, warm_start=True)
rfc.fit(x_train_1, y_train_1)
y_pred = rfc.predict(x_test)
test_score = accuracy = accuracy_score(y_test, y_pred)
print("Testing Score", test_score)

# Let's say we have more training data available and we want to use this
# training data to further train the model
rfc.n_estimators += 2
rfc.fit(x_train_2, y_train_2)
y_pred = rfc.predict(x_test)
test_score = accuracy = accuracy_score(y_test, y_pred)
print("Testing Score", test_score)