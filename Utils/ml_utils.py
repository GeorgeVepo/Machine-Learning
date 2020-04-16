from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import Binarizer, KBinsDiscretizer
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from statsmodels.stats.outliers_influence import variance_inflation_factor
import numpy as np
import pandas as pd
from enum import Enum


desired_width = 320
pd.set_option('display.width', desired_width)
np.set_printoptions(linewidth=desired_width)
pd.set_option('display.max_columns', 10)


class TransformMethod(Enum):
    none = 0
    min_max_scaler = 1
    standard_scaler = 2
    normalize_l1 = 3
    normalize_l2 = 4
    normalize_max = 5
    binarize = 6


def univariable_linear_regression(features, x_label, y_label, test_frac=0.2, discretization=False, n_bins=0, normalize=True):
    y = np.ravel(features[[y_label]])
    x = features[[x_label]]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_frac)
    if discretization:
        enc = KBinsDiscretizer(n_bins=n_bins, encode='ordinal')
        enc.fit(x_train),
        x_train = enc.transform(x_train)
        x_test = enc.transform(x_test)

    linear_model = LinearRegression(normalize=normalize).fit(x_train, y_train)
    y_pred = linear_model.predict(x_test)
    print('Testing score', r2_score(y_test, y_pred).__round__(5))


def linear_regression(features, label, test_frac=0.2, discretization=False, n_bins=0, normalize=True, adjust_r2=True):
    x, y = transform_data(features, label, TransformMethod.none)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_frac)
    linear_model = LinearRegression(normalize=normalize).fit(x_train, y_train)
    if discretization:
        enc = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy='uniform')
        enc.fit(x_train)
        x_test = enc.transform(x_test)

    y_pred = linear_model.predict(x_test)
    print('Testing score', r2_score(y_test, y_pred).__round__(5))
    if adjust_r2:
        print('Adjusted r2 score : ', adjusted_r2(r2_score(y_test, y_pred), y_test, x_test).__round__(5))
        print('Difference of scores: ', (adjusted_r2(r2_score(y_test, y_pred), y_test, x_test) - r2_score(y_test, y_pred)).__round__(5) * -1)


def logistic_regression(features, label, transform_method, test_frac=0.2, solver='liblinear'):
    x, y = transform_data(features, label, transform_method)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_frac)
    model = LogisticRegression(solver=solver).fit(x_train, y_train)
    y_pred = model.predict(x_test)
    print("Testing score : ", accuracy_score(y_test, y_pred).__round__(5))


def linear_regression_model(features, label, test_frac=0.2, normalize=True):
    x, y = transform_data(features, label, TransformMethod.none)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_frac)
    return LinearRegression(normalize=normalize).fit(x_train, y_train)


def logistic_regression_model(features, label, transform_method, test_frac=0.2, solver='liblinear'):
    x, y = transform_data(features, label, transform_method)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_frac)
    return LogisticRegression(solver=solver).fit(x_train, y_train.values.ravel())


def transform_data(features, label, method):
    features, label = set_apart(features, label)

    if method == TransformMethod.min_max_scaler:
        features = min_max_scaler(features)

    elif method == TransformMethod.standard_scaler:
        features = standard_scaler(features)

    elif method == TransformMethod.normalize_l1:
        features = normalize_l1(features)

    elif method == TransformMethod.normalize_l2:
        features = normalize_l2(features)

    elif method == TransformMethod.normalize_max:
        features = normalize_max(features)

    elif method == TransformMethod.binarize:
        features = binarize(features)

    label = np.ravel(label)
    return features, label


def min_max_scaler(features):
    scaler = MinMaxScaler(feature_range=(0, 1))
    rescaled_features = scaler.fit_transform(features)
    return pd.DataFrame(rescaled_features, columns=features.columns)


def standard_scaler(features):
    scaler = StandardScaler()
    scaler.fit(features)
    standardized_features = scaler.transform(features)
    return pd.DataFrame(standardized_features, columns=features.columns)


def normalize_l1(features):
    normalizer = Normalizer(norm='l1')
    normalized_features = normalizer.fit_transform(features)
    return pd.DataFrame(normalized_features, columns=features.columns)


def normalize_l2(features):
    normalizer = Normalizer(norm='l2')
    normalized_features = normalizer.fit_transform(features)
    return pd.DataFrame(normalized_features, columns=features.columns)


def normalize_max(features):
    normalizer = Normalizer(norm='max')
    normalized_features = normalizer.fit_transform(features)
    return pd.DataFrame(normalized_features, columns=features.columns)


def binarize(features):
    data = None
    for i in range(0, features.shape[1]):
        scaler = Binarizer(threshold=float((features[[features.columns[i]]]).mean())).fit(features[[features.columns[i]]])
        new_binarized_feature = scaler.transform(features[[features.columns[i]]])
        if data is None:
            data = new_binarized_feature
        else:
            data = np.concatenate((data, new_binarized_feature), axis=1)

    return pd.DataFrame(data, columns=features.columns)


def set_apart(features, label):
    return features.drop(label, axis=1), features[[label]]


def get_columns_to_end(features, begin):
    data = None
    for i in range(begin, features.shape[1]):
        if data is not None:
            data = pd.concat([data, features[features.columns[i]]], axis=1)
        else:
            data = features[features.columns[i]]

    print(data.sample(10))
    return data


def get_columns(features, begin, end):
    data = None
    for i in range(begin, end):
        if data is not None:
            pd.concat([data, features[features.columns[i]]], axis=1)
        else:
            data = features[features.columns[i]]

    print(data.sample(10))
    return data


# Configure the print command to show more data
def config_print():
    desired_width1 = 320
    pd.set_option('display.width', desired_width1)
    np.set_printoptions(linewidth=desired_width1)
    pd.set_option('display.max_columns', 10)


def adjusted_r2(r_square, labels, features):
    adj_r_square = 1 - ((1 - r_square) * (len(labels) - 1)) / (len(labels) - features.shape[1] - 1)
    return adj_r_square


def variance_inflation_factor_calculator(features):
    vif = pd.DataFrame()
    vif['Features'] = features.columns
    vif['VIF Factor'] = [variance_inflation_factor(features.values, i) for i in range(features.shape[1])]
    print(vif.round(2))
    return vif
