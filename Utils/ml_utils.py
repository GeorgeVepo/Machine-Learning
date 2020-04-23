from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, chi2, f_classif, SelectPercentile, RFE
from sklearn.impute._iterative import IterativeImputer
from sklearn.linear_model import LogisticRegression, Lasso
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import Binarizer, KBinsDiscretizer
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score
from sklearn.tree import DecisionTreeRegressor
from statsmodels.stats.outliers_influence import variance_inflation_factor
from mlxtend.feature_selection import SequentialFeatureSelector
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


class FeatureSelectionMethod(Enum):
    none = 0
    missing_value_ratio = 1
    chi2 = 2
    anova = 3
    rfe = 4
    forward = 5
    backward = 6
    lasso = 7
    decision_tree = 8


def univariable_linear_regression(features, x_label, y_label, test_frac=0.2, discretization=False, n_bins=0, normalize=True, adjust_r2=True):
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
    print('Testing score', r2_score(y_test, y_pred))
    if adjust_r2:
        print('Adjusted r2 score : ', adjusted_r2(r2_score(y_test, y_pred), y_test, x_test))
        print('Difference of scores: ', (adjusted_r2(r2_score(y_test, y_pred), y_test, x_test) - r2_score(y_test, y_pred)) * -1)


def linear_regression(features, label, test_frac=0.2, discretization=False, n_bins=0, normalize=True, adjust_r2=True, transform_method=TransformMethod.none, feature_selection_method=FeatureSelectionMethod.none):
    model = LinearRegression(normalize=normalize)
    x, y = transform_data(features, label, transform_method, feature_selection_method, model)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_frac)
    if discretization:
        enc = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy='uniform')
        enc.fit(x_train)
        x_train = enc.transform(x_train)
        x_test = enc.transform(x_test)

    linear_model = model.fit(x_train, y_train)
    y_pred = linear_model.predict(x_test)
    print('Testing score', r2_score(y_test, y_pred))
    if adjust_r2:
        print('Adjusted r2 score : ', adjusted_r2(r2_score(y_test, y_pred), y_test, x_test))
        print('Difference of scores: ', (adjusted_r2(r2_score(y_test, y_pred), y_test, x_test) - r2_score(y_test, y_pred)) * -1)


def logistic_regression(features, label, test_frac=0.2, solver='liblinear',transform_method=TransformMethod.none, feature_selection_method=FeatureSelectionMethod.none):
    model = LogisticRegression(solver=solver)
    x, y = transform_data(features, label, transform_method, feature_selection_method, model)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_frac)
    model = model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    print("Testing score : ", accuracy_score(y_test, y_pred))


def linear_regression_model(features, label, test_frac=0.2, normalize=True, transform_method=TransformMethod.none, feature_selection_method=FeatureSelectionMethod.none):
    model = LinearRegression(normalize=normalize)
    x, y = transform_data(features, label, transform_method, feature_selection_method, model)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_frac)
    return model.fit(x_train, y_train), x_test, y_test


def logistic_regression_model(features, label, test_frac=0.2, solver='liblinear', transform_method=TransformMethod.none, feature_selection_method=FeatureSelectionMethod.none):
    model = LogisticRegression(solver=solver)
    x, y = transform_data(features, label, transform_method, feature_selection_method, model)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_frac)
    return model.fit(x_train, y_train), x_test, y_test


def transform_data(features, label, transformation_method, feature_selection_method, model):
    features, label = set_apart(features, label)
    features = feature_selection(features, label, feature_selection_method, model)

    if transformation_method == TransformMethod.min_max_scaler:
        features = min_max_scaler(features)

    elif transformation_method == TransformMethod.standard_scaler:
        features = standard_scaler(features)

    elif transformation_method == TransformMethod.normalize_l1:
        features = normalize_l1(features)

    elif transformation_method == TransformMethod.normalize_l2:
        features = normalize_l2(features)

    elif transformation_method == TransformMethod.normalize_max:
        features = normalize_max(features)

    elif transformation_method == TransformMethod.binarize:
        features = binarize(features)

    label = np.ravel(label)
    return features, label


def univariate_feature_imputation(feature, strategy):
    imp = SimpleImputer(missing_values=np.nan, strategy=strategy)
    imp.fit(feature.values.reshape(-1, 1))
    return imp.transform(feature.values.reshape(-1, 1))


def multivariate_feature_imputation(features):
    imp = IterativeImputer(max_iter=100, random_state=0)
    imp.fit(features)
    feature_arr = imp.transform(features)
    return pd.DataFrame(feature_arr, columns=features.columns)


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
    return features.drop(label, axis=1), features[label]


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


def feature_selection(features, label, method, model):
    if method == FeatureSelectionMethod.missing_value_ratio:
        features = feature_selection_missing_value_ratio(features)
        features = multivariate_feature_imputation(features)
        return features
    if method == FeatureSelectionMethod.chi2:
        features = feature_selection_chi2(features, label)
        features = multivariate_feature_imputation(features)
        return features
    if method == FeatureSelectionMethod.anova:
        features = feature_selection_anova(features, label)
        features = multivariate_feature_imputation(features)
        return features
    if method == FeatureSelectionMethod.rfe:
        features = feature_selection_rfe(features, label, model)
        features = multivariate_feature_imputation(features)
        return features
    if method == FeatureSelectionMethod.forward:
        features = feature_selection_forward(features, label)
        features = multivariate_feature_imputation(features)
        return features
    if method == FeatureSelectionMethod.backward:
        features = feature_selection_backward(features, label)
        features = multivariate_feature_imputation(features)
        return features
    if method == FeatureSelectionMethod.lasso:
        features = feature_selection_lasso(features, label)
        features = multivariate_feature_imputation(features)
        return features
    if method == FeatureSelectionMethod.decision_tree:
        features = feature_selection_decision_tree(features, label)
        features = multivariate_feature_imputation(features)
        return features

    features = multivariate_feature_imputation(features)
    return features


def feature_selection_missing_value_ratio(features):
    return features.dropna(thresh=int(features.shape[0] * .9), axis=1)


def feature_selection_chi2(features, label):
    x = pd.DataFrame(features).fillna(0)
    fit = SelectKBest(score_func=chi2, k=4).fit(x.astype(np.float64), label)
    x_new = fit.transform(features.fillna(0).astype(np.float64))
    return features[get_selected_features(x, pd.DataFrame(x_new))]


def feature_selection_anova(features, label):
    x = pd.DataFrame(features).fillna(0)
    fit = SelectPercentile(f_classif, percentile=80).fit(x.astype(np.float64), label)
    x_new = fit.transform(features.fillna(0).astype(np.float64))
    return features[get_selected_features(x, pd.DataFrame(x_new))]


def feature_selection_rfe(features, label, model):
    fit = RFE(model, n_features_to_select=4).fit(features, label)
    feature_rank = pd.DataFrame({'columns': features.columns,
                                 'ranking': fit.ranking_,
                                 'selected': fit.support_})
    recursive_feature_names = feature_rank.loc[feature_rank['selected'] == True]

    return features[recursive_feature_names['columns'].values]


def feature_selection_forward(features, label):
    feature_selector = SequentialFeatureSelector(RandomForestClassifier(n_estimators=10),
                                    k_features=4,
                                    forward=True,
                                    scoring='accuracy',
                                    cv=4)

    feature_selector.fit(np.array(features), label)
    names = list(features.columns[list(feature_selector.k_feature_idx_)])
    return features[names]


def feature_selection_backward(features, label):
    feature_selector = SequentialFeatureSelector(RandomForestClassifier(n_estimators=10),
                                    k_features=4,
                                    forward=False,
                                    scoring='accuracy',
                                    cv=4)

    feature_selector.fit(np.array(features), label)
    names = list(features.columns[list(feature_selector.k_feature_idx_)])
    return features[names]


def feature_selection_lasso(features, label):
    lasso = Lasso(alpha=0.8)
    lasso.fit(features, label)
    coef = pd.Series(lasso.coef_, features.columns).sort_values().head(2)
    return features[list(coef.keys())]


def feature_selection_decision_tree(features, label):
    decision_tree = DecisionTreeRegressor(max_depth=4)
    decision_tree.fit(features, label)
    coef = pd.Series(decision_tree.feature_importances_, features.columns).sort_values(ascending=False).head(2)
    return features[list(coef.keys())]


def get_selected_features(x, x_new):
    selected_features = []

    for i in range(len(x_new.columns)):
        for j in range(len(x.columns)):
            if x_new.iloc[:, i].equals(x.iloc[:, j]):
                selected_features.append((x.columns[j]))

    return selected_features
