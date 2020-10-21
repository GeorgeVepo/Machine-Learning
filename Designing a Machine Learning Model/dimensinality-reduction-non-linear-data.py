import pandas as pd
import seaborn as sns

from sklearn import datasets

import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from sklearn.manifold import MDS, LocallyLinearEmbedding

# Generate artificial data. In this case, non-linear data which is shaped in the form od a swiss roll


x, color = datasets._samples_generator.make_swiss_roll(n_samples=2000)

# Plotting the data

x = pd.DataFrame(x)

ax = plt.subplots(figsize=(8, 8))
ax = plt.axes(projection='3d')

ax.scatter3D(x[0], x[1], x[2], c=color, cmap=plt.cm.Spectral)
plt.show()

def apply_manifold_learning(feature, method):
    feature = method.fit_transform(feature)
    print("Now Shape of X : ", feature.shape)
    feature = pd.DataFrame(feature)
    plt.subplots(figsize=(8, 8))
    plt.axis('equal')

    plt.scatter(feature[0], feature[1], c=color, cmap=plt.cm.Spectral)
    plt.xlabel('X[0]')
    plt.xlabel('X[1]')
    plt.show()
    return method


# Reduces dimensionality while trying to preserve
# the distances between instances
mds = apply_manifold_learning(x, MDS(n_components=2, metric=True))

# Measures how each instance relates to its closest neighbors
# and tries to find a lower dimensionality representation which
# preserves these local relationships
lle = apply_manifold_learning(x, LocallyLinearEmbedding(n_neighbors=15,
                                                        n_components=2,
                                                        method='hessian'))




