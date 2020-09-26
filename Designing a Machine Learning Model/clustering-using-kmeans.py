import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

drivers_data = pd.read_csv('dataset/driver_dataset.csv', sep='\t')

drivers_data = drivers_data.sample(frac=1)
# Drop useless column
drivers_data.drop('Driver_ID', axis=1, inplace=True)
drivers_data.sample(10)

# group in 4 clusters
kmeans_model = KMeans(n_clusters=4, max_iter=1000).fit(drivers_data)

print(np.unique(kmeans_model.labels_))

zipped_list = list(zip(np.array(drivers_data), kmeans_model.labels_))

centroids = kmeans_model.cluster_centers_

colors = ['g', 'y', 'b', 'c']
plt.figure(figsize=(5, 4))

for element in zipped_list:
    plt.scatter(element[0][0], element[0][1],
                c=colors[(element[1] % len(colors))])

plt.scatter(centroids[:, 0], centroids[:, 1], c='r', s=200, marker='s')

for i in range(len(centroids)):
    plt.annotate(i, (centroids[i][0], centroids[i][1]), fontsize=20)

plt.show()


# How similar an object is compared with other
# objects in its own cluster (cohesion) and how different
# it is from objects in other clusters (separation).
# Observation: Change the number of clusters can improve the score.
print("Silhouette score", silhouette_score(drivers_data, kmeans_model.labels_))

