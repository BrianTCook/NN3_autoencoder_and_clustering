import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

#load and prepare data
data = np.array(pd.read_csv('encoded_with_labels.txt', sep = ',', header=None))
encoded_values = data[:,2:4]

'''
labels = np.array([])
for i in range(len(data)):
    labels = np.append(labels, data[i,0] + ' ' + str(data[i,1]))

plt.scatter(encoded_values[:,0], encoded_values[:,1], s=3)
plt.show()
#plt.close()
'''

ns = [2, 5, 20]

plt.rc('text', usetex = True)
plt.rc('font', family = 'serif')

for n in ns:

    plt.figure()
    
    #apply kmeans clustering and plot
    kmeans = KMeans(n_clusters=n)
    kmeans.fit(encoded_values)
    y_kmeans = kmeans.predict(encoded_values)
    centers = kmeans.cluster_centers_
    plt.scatter(encoded_values[:, 0], encoded_values[:, 1], c=y_kmeans, s=5, cmap='Dark2')
    #plt.text(encoded_values[:, 0], encoded_values[:, 1], labels, fontsize=8)
    plt.xlabel(r'$x$', fontsize=16)
    plt.ylabel(r'$y$', fontsize=16)
    plt.title(r'$k$-means, $N_{clusters} = %i$'%n, fontsize=16)
    plt.savefig('kmeans_nclusters=%i.pdf'%n)

'''
#show scatter with all labels
for i,label in enumerate(labels):
    x = encoded_values[:,0][i]
    y = encoded_values[:,1][i]
    plt.scatter(x, y, s=4)
    #plt.text(x, y, label, fontsize=5)
plt.show()
'''
