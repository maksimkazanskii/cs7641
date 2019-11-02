# https://stackoverflow.com/questions/32750915/pca-inverse-transform-manually
#
import numpy as np
import pandas as pd
import os
import time
import random
import numpy as np
from sklearn import preprocessing
from sklearn.datasets import make_blobs
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
import os
from sklearn.metrics import silhouette_samples, silhouette_score
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import colorConverter
import matplotlib.cm as cm
from sklearn.cluster import KMeans
from sklearn import metrics
import data_prep as data
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import datasets, cluster

X_full,Y_full = data.get_data_MNIST()
X_full = preprocessing.normalize(X_full, norm='l2', axis=1)
X, X_valtest, Y, Y_valtest = train_test_split(X_full, Y_full, test_size=0.4, random_state =111)

#X_full,Y_full = get_data_heart()
#print("X shape", X_full.shape)
#print("Y shape", Y_full.shape)


X_safe = np.copy(X)
Y_safe = np.copy(Y)




np.random.seed(111)
range_n_clusters = np.arange(2,40,1)
SILHOUETTES = []

for n_clusters in range_n_clusters:
    # Create a subplot with 1 row and 2 columns


    # Initialize the clusterer with n_clusters value and a random generator
    # seed of 10 for reproducibility.
    clusterer = KMeans(n_clusters=n_clusters, random_state=1)
    cluster_labels = clusterer.fit_predict(X)

    # The silhouette_score gives the average value for all the samples.
    # This gives a perspective into the density and separation of the formed
    # clusters
    silhouette_avg = silhouette_score(X, cluster_labels)
    print("For n_clusters =", n_clusters,
          "The average silhouette_score is :", silhouette_avg)
    SILHOUETTES.append(silhouette_avg)


np.save("data/kmeans_main_sils.npy",SILHOUETTES)

SILHOUETTES = np.load('data/kmeans_main_sils.npy', allow_pickle=True)



n_clusters =10
clusterer = KMeans(n_clusters=n_clusters, random_state=1)
cluster_labels = clusterer.fit_predict(X)
silhouette_avg = silhouette_score(X, cluster_labels)
fig, ax1, = plt.subplots()
# The 1st subplot is the silhouette plot
# The silhouette coefficient can range from -1, 1 but in this example all
# lie within [-0.1, 1]

# The (n_clusters+1)*10 is for inserting blank space between silhouette
# plots of individual clusters, to demarcate them clearly.
ax1.set_ylim([0, len(X) + (n_clusters + 1) * 10])
# Compute the silhouette scores for each sample
sample_silhouette_values = silhouette_samples(X, cluster_labels)
y_lower = 10
for i in range(n_clusters):
    # Aggregate the silhouette scores for samples belonging to
    # cluster i, and sort them
    ith_cluster_silhouette_values = \
                sample_silhouette_values[cluster_labels == i]

    ith_cluster_silhouette_values.sort()

    size_cluster_i = ith_cluster_silhouette_values.shape[0]
    y_upper = y_lower + size_cluster_i

    color = cm.nipy_spectral(float(i) / n_clusters)
    ax1.fill_betweenx(np.arange(y_lower, y_upper),
                          0, ith_cluster_silhouette_values,
                          facecolor=color, edgecolor=color, alpha=0.7)

    # Label the silhouette plots with their cluster numbers at the middle
    ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

    # Compute the new y_lower for next plot
    y_lower = y_upper + 10  # 10 for the 0 samples
plt.xlim([-0.1, 0.5])
ax1.set_title(("The silhouette plot for the various clusters."), fontsize = 16)
ax1.set_xlabel("The silhouette coefficient values")
ax1.set_ylabel("Cluster label")

# The vertical line for average silhouette score of all the values
ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

ax1.set_yticks([])  # Clear the yaxis labels / ticks
ax1.set_xticks([-0.1, 0, 0.1, 0.2, 0.3, 0.4, 0.5])

# 2nd Plot showing the actual clusters formed
colors = cm.nipy_spectral(cluster_labels.astype(float) / n_clusters)
plt.title(("Silhouette analysis for KMeans clustering \n on sample data "
                  "with n_clusters = %d" % n_clusters))
filename = "plots/mnist_kmeans_silljoutes_"+ str(n_clusters)+".png"
plt.savefig(filename)




max_value=10
print(range_n_clusters[max_value-2], SILHOUETTES[max_value-2])
fig, ax = plt.subplots()
plt.plot(range_n_clusters , SILHOUETTES , 'o', color = 'steelblue')
plt.plot(range_n_clusters, SILHOUETTES  , '-', color = 'steelblue', alpha = 0.5)
# !!!! fix the optimal value
plt.plot(range_n_clusters[max_value-2], SILHOUETTES[max_value-2], 'ro')

plt.title("Sillhouettes score vs number of clusters  (KMeans)")
plt.xlabel('Number of Clusters') , plt.ylabel('Sillhouettes score')
plt.savefig("plots/mnist_kmeans_sillhouttes.png")

# pca

X = np.copy(X_safe)
Y = np.copy(Y_safe)
# Comparing with True labels

inertia = []
ARI = []
ARI_ARR = np.arange(2,40,1)

for n in ARI_ARR:
    algorithm = KMeans(n_clusters = n ,init='k-means++', n_init = 30 ,max_iter=1000,
                        tol=0.0001,  random_state= 111  , algorithm='elkan')

    algorithm.fit(X)
    inertia.append(algorithm.inertia_)
    labels_pred = algorithm.labels_
    print("n_clusters, ", n)
    ARI.append(metrics.adjusted_rand_score(Y, labels_pred))



np.save("data/kmeans_main_ARI.npy",ARI)

ARI = np.load('data/kmeans_main_ARI.npy', allow_pickle=True)
max_value =14

fig, ax = plt.subplots()
plt.plot(ARI_ARR, ARI , 'o', color = 'steelblue')
plt.plot(ARI_ARR, ARI , '-', color = 'steelblue' , alpha = 0.5)
plt.plot(ARI_ARR[max_value-2], ARI[max_value-2], 'ro')
plt.title("ARI (K-Means algorithm). ")
plt.xlabel('Number of Clusters') , plt.ylabel('ARI')
plt.savefig("plots/mnist_kmeans_ARI.png")

#
# PCA
#


X = np.copy(X_safe)
Y = np.copy(Y_safe)
scaler = StandardScaler()
x = scaler.fit_transform(X)
print("PCA analysis")
from sklearn.decomposition import PCA
pca = PCA(n_components=70)
pca.fit(X)
eigenvalues = pca.singular_values_
ratios      = pca.explained_variance_ratio_

fig, ax = plt.subplots()
plt.plot(np.arange(1,len(eigenvalues)+1,1) , eigenvalues , 'o', color = 'steelblue', markersize=2)
plt.plot(np.arange(1,len(eigenvalues)+1,1) , eigenvalues , '-', color = 'steelblue' , alpha = 0.5)
plt.title("Eigenvalues, PCA")
plt.xlabel('Component number') , plt.ylabel('Eigenvalue')
plt.savefig("plots/MNIST_pca_eigenvalues.png")

ratios = np.cumsum(ratios)
fig, ax = plt.subplots()
plt.plot(np.arange(1,len(ratios)+1,1) , ratios , 'o', color = 'steelblue', markersize = 2 )
plt.plot(np.arange(1,len(ratios)+1,1) , ratios , '-', color = 'steelblue' , alpha = 0.5)
plt.axhline(y=0.8, color='r', linestyle='--')
plt.axhline(y=0.6, color='r', linestyle='--')
plt.title("Cummulative variance, PCA")
plt.xlabel('Component number') , plt.ylabel('Cummulative variance')
plt.savefig("plots/MNIST_pca_cum_variance.png")


pca = PCA(n_components=45)
pca.fit(X)
X = pca.transform(X)

range_n_clusters = np.arange(2,40,1)
SILHOUETTES = []



for n_clusters in range_n_clusters:
    # Create a subplot with 1 row and 2 columns

    # Initialize the clusterer with n_clusters value and a random generator
    # seed of 10 for reproducibility.
    clusterer = KMeans(n_clusters=n_clusters, random_state=1)
    cluster_labels = clusterer.fit_predict(X)

    # The silhouette_score gives the average value for all the samples.
    # This gives a perspective into the density and separation of the formed
    # clusters
    silhouette_avg = silhouette_score(X, cluster_labels)
    print("For n_clusters =", n_clusters,
          "The average silhouette_score is :", silhouette_avg)
    SILHOUETTES.append(silhouette_avg)


np.save("data/kmeans_PCA_sils.npy",SILHOUETTES)

SILHOUETTES = np.load('data/kmeans_PCA_sils.npy', allow_pickle=True)

max_value=24
print(range_n_clusters[max_value-2], SILHOUETTES[max_value-2])
fig, ax = plt.subplots()
plt.plot(range_n_clusters , SILHOUETTES , 'o', color = 'steelblue')
plt.plot(range_n_clusters, SILHOUETTES  , '-', color = 'steelblue', alpha = 0.5)
# !!!! fix the optimal value
plt.plot(range_n_clusters[max_value-2], SILHOUETTES[max_value-2], 'ro')
plt.title("Sillhouettes score vs number of clusters  (KMeans, PCA)")
plt.xlabel('Number of Clusters') , plt.ylabel('Sillhouettes score')
plt.savefig("plots/mnist_kmeans_PCA_sils.png")


inertia = []
ARI = []
ARI_ARR = np.arange(2,40,1)

for n in ARI_ARR:
    algorithm = KMeans(n_clusters = n ,init='k-means++', n_init = 30 ,max_iter=1000,
                        tol=0.0001,  random_state= 111  , algorithm='elkan')
    algorithm.fit(X)
    inertia.append(algorithm.inertia_)
    labels_pred = algorithm.labels_
    print("n_clusters, ", n)
    ARI.append(metrics.adjusted_rand_score(Y, labels_pred))

np.save("data/kmeans_PSA_ARI.npy",ARI)

ARI = np.load('data/kmeans_PSA_ARI.npy', allow_pickle=True)
max_value =15

fig, ax = plt.subplots()
plt.plot(ARI_ARR, ARI , 'o', color = 'steelblue')
plt.plot(ARI_ARR, ARI , '-', color = 'steelblue' , alpha = 0.5)
plt.plot(ARI_ARR[max_value-2], ARI[max_value-2], 'ro')
plt.title("ARI (K-Means algorithm ,PCA). ")
plt.xlabel('Number of Clusters') , plt.ylabel('ARI')
plt.savefig("plots/mnist_kmeans_PCA_ARI.png")