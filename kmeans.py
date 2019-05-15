import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets.samples_generator import make_blobs

#load and prepare data
data = np.array(pd.read_csv('encoded_with_labels.txt', sep = ',', header=None))
encoded_values = data[:,2:4]

labels = np.array([])
for i in range(len(data)):
    labels = np.append(labels, data[i,0] + ' ' + str(data[i,1]))

plt.scatter(encoded_values[:,0], encoded_values[:,1], s=3)
plt.show()
plt.close()

#apply kmeans clustering and plot
kmeans = KMeans(n_clusters=)
kmeans.fit(encoded_values)
y_kmeans = kmeans.predict(encoded_values)
centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='black', s=50, alpha=0.5);
plt.scatter(encoded_values[:, 0], encoded_values[:, 1], c=y_kmeans, s=5, cmap='Dark2')
plt.text(encoded_values[:, 0], encoded_values[:, 1], labels, fontsize=8)
plt.show()

#show scatter with all labels
for i,label in enumerate(labels):
    x = encoded_values[:,0][i]
    y = encoded_values[:,1][i]
    plt.scatter(x, y, s=4)
    plt.text(x, y, label, fontsize=5)
plt.show()
