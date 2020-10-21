# Performing dimensionality reduction on linear data
# using Principal Component Analysis (PCA)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

diabetes_data = pd.read_csv('dataset/PimaIndians_processed.csv')

FEATURES = list(diabetes_data.columns[:-1])

def apply_pca(n_components, features):
    pca = PCA(n_components=n_components)
    x_new = pca.fit_transform(features)
    print("Explained Variance:", pca.explained_variance_ratio_)
    return pca, pd.DataFrame(x_new)


pca_obj, _ = apply_pca(4, diabetes_data[FEATURES])


