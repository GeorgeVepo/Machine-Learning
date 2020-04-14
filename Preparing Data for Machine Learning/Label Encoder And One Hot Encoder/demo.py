from sklearn import preprocessing
from ml_utils import get_columns_to_end
import pandas as pd

gosales_df = pd.read_csv('../datasets/GoSales_Tx_LogisticRegression.csv')

gender = ['M', 'F']

label_encoding = preprocessing.LabelEncoder()
label_encoding = label_encoding.fit(gender)
gosales_df['GENDER'] = label_encoding.transform(gosales_df['GENDER'].astype(str))

one_hot_encoding = preprocessing.OneHotEncoder()
one_hot_encoding = one_hot_encoding.fit(gosales_df['MARITAL_STATUS'].values.reshape(-1, 1))
one_hot_encoding_label = one_hot_encoding.transform(gosales_df['MARITAL_STATUS'].values.reshape(-1, 1)).toarray()
label_df = pd.DataFrame()

label_df['MARITAL_STATUS_Married'] = one_hot_encoding_label[:, 0]
label_df['MARITAL_STATUS_Single'] = one_hot_encoding_label[:, 1]
label_df['MARITAL_STATUS_Unspecified'] = one_hot_encoding_label[:, 2]

gosales_df = pd.concat([gosales_df, label_df], axis=1)
gosales_df = gosales_df.drop('MARITAL_STATUS', axis=1)

get_columns_to_end(gosales_df, gosales_df.shape[1] - 15)
gosales_df = pd.get_dummies(gosales_df, columns=['PROFESSION'])
get_columns_to_end(gosales_df, gosales_df.shape[1] - 15)
