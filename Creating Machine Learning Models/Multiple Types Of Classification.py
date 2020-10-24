import pandas as pd
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

utils.naive_bayes(data, 'customer_type')
utils.k_nearest_neighbors(data, 'customer_type')
utils.svc(data, 'customer_type')
utils.decision_tree_classifier(data, 'customer_type', max_depth=6)
