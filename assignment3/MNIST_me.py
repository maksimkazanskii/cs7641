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
from sklearn.mixture import GaussianMixture

X_full,Y_full = data.get_data_MNIST()
X_full = preprocessing.normalize(X_full, norm='l2', axis=1)
X, X_valtest, Y, Y_valtest = train_test_split(X_full, Y_full, test_size=0.4, random_state =111)

#X_full,Y_full = get_data_heart()
#print("X shape", X_full.shape)
#print("Y shape", Y_full.shape)


X_safe = np.copy(X)
Y_safe = np.copy(Y)
print(X_safe.shape)


np.random.seed(111)
range_n_clusters = np.arange(2,40,1)
SILHOUETTES = []

for n_clusters in range_n_clusters:
    # Create a subplot with 1 row and 2 columns

    # Initialize the clusterer with n_clusters value and a random generator
    # seed of 10 for reproducibility.

    clusterer =GaussianMixture(n_components=n_clusters, random_state=11, covariance_type='full')

    clusterer.fit(X)
    labels = clusterer.predict(X)
    #inertia.append(algorithm.inertia_)
    cluster_labels = clusterer.fit_predict(X)

    # The silhouette_score gives the average value for all the samples.
    # This gives a perspective into the density and separation of the formed
    # clusters
    silhouette_avg = silhouette_score(X, cluster_labels)
    print("For n_clusters =", n_clusters,
          "The average silhouette_score is :", silhouette_avg)
    SILHOUETTES.append(silhouette_avg)


np.save("data/me_main_sils.npy",SILHOUETTES)

SILHOUETTES = np.load('data/me_main_sils.npy', allow_pickle=True)



n_clusters =12
clusterer = GaussianMixture(n_components=n_clusters, random_state=11, covariance_type='full')
clusterer.fit(X)
labels = clusterer.predict(X)
# inertia.append(algorithm.inertia_)
cluster_labels = clusterer.fit_predict(X)
silhouette_avg = silhouette_score(X, cluster_labels)
fig, ax1, = plt.subplots()
# The 1st subplot is the silhouette plot
# The silhouette coefficient can range from -1, 1 but in this example all
# lie within [-0.1, 1]
ax1.set_xlim([-0.1, 0.5])
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

ax1.set_title(("The silhouette plot for the various clusters."), fontsize = 16)
ax1.set_xlabel("The silhouette coefficient values")
ax1.set_ylabel("Cluster label")

# The vertical line for average silhouette score of all the values
ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

ax1.set_yticks([])  # Clear the yaxis labels / ticks
ax1.set_xticks([-0.1, 0, 0.1, 0.2, 0.3, 0.4, 0.5])

# 2nd Plot showing the actual clusters formed
colors = cm.nipy_spectral(cluster_labels.astype(float) / n_clusters)
plt.title(("Silhouette analysis for Expectation maximization clustering \n on sample data "
                  "with n_clusters = %d" % n_clusters))
filename = "plots/mnist_me_silljoutes_"+ str(n_clusters)+".png"
plt.savefig(filename)




max_value=12
print(range_n_clusters[max_value-2], SILHOUETTES[max_value-2])
fig, ax = plt.subplots()
plt.plot(range_n_clusters , SILHOUETTES , 'o', color = 'steelblue')
plt.plot(range_n_clusters, SILHOUETTES  , '-', color = 'steelblue', alpha = 0.5)
# !!!! fix the optimal value
plt.plot(range_n_clusters[max_value-2], SILHOUETTES[max_value-2], 'ro')

plt.title("Sillhouettes score vs number of clusters \n(expetation maximization)")
plt.xlabel('Number of Clusters') , plt.ylabel('Sillhouettes score')
plt.savefig("plots/mnist_me_sillhouttes.png")

# pca

X = np.copy(X_safe)
Y = np.copy(Y_safe)
# Comparing with True labels

inertia = []
ARI = []
ARI_ARR = np.arange(2,60,1)

for n in ARI_ARR:

    clusterer = GaussianMixture(n_components=n, random_state=111, covariance_type='full')
    clusterer.fit(X)
    labels_pred = clusterer.predict(X)

    # inertia.append(algorithm.inertia_)
    ari = metrics.adjusted_rand_score(Y, labels_pred)
    print("n_clusters, ", n)
    ARI.append(ari)

np.save("data/me_main_ARI.npy",ARI)

ARI = np.load('data/me_main_ARI.npy', allow_pickle=True)
print(ARI_ARR, ARI)
max_value =16

fig, ax = plt.subplots()
plt.plot(ARI_ARR, ARI , 'o', color = 'steelblue')
plt.plot(ARI_ARR, ARI , '-', color = 'steelblue' , alpha = 0.5)
plt.plot(ARI_ARR[max_value-2], ARI[max_value-2], 'ro')
plt.title("ARI (Expectation maximization). ")
plt.xlabel('Number of Clusters') , plt.ylabel('ARI')
plt.savefig("plots/mnist_me_ARI.png")

#
# PCA
#
print("PCA analysis")
from sklearn.decomposition import PCA
pca = PCA(n_components=45)
pca.fit(X)
X = pca.transform(X)

SILHOUETTES = []
for n_clusters in range_n_clusters:
    # Create a subplot with 1 row and 2 columns

    # Initialize the clusterer with n_clusters value and a random generator
    # seed of 10 for reproducibility.

    clusterer =GaussianMixture(n_components=n_clusters, random_state=11, covariance_type='full')

    clusterer.fit(X)
    labels = clusterer.predict(X)
    #inertia.append(algorithm.inertia_)
    cluster_labels = clusterer.fit_predict(X)

    # The silhouette_score gives the average value for all the samples.
    # This gives a perspective into the density and separation of the formed
    # clusters
    silhouette_avg = silhouette_score(X, cluster_labels)
    print("For n_clusters =", n_clusters,
          "The average silhouette_score is :", silhouette_avg)
    SILHOUETTES.append(silhouette_avg)


np.save("data/me_pca_sils.npy",SILHOUETTES)

SILHOUETTES = np.load('data/me_pca_sils.npy', allow_pickle=True)


max_value=12
print(range_n_clusters[max_value-2], SILHOUETTES[max_value-2])
fig, ax = plt.subplots()
plt.plot(range_n_clusters , SILHOUETTES , 'o', color = 'steelblue')
plt.plot(range_n_clusters, SILHOUETTES  , '-', color = 'steelblue', alpha = 0.5)
# !!!! fix the optimal value
plt.plot(range_n_clusters[max_value-2], SILHOUETTES[max_value-2], 'ro')

plt.title("Sillhouettes score vs number of clusters \n(expetation maximization), PCA")
plt.xlabel('Number of Clusters') , plt.ylabel('Sillhouettes score')
plt.savefig("plots/mnist_me_pca_sils.png")

inertia = []
ARI = []
ARI_ARR = np.arange(2,40,1)

for n in ARI_ARR:

    clusterer = GaussianMixture(n_components=n, random_state=111, covariance_type='full')
    clusterer.fit(X)
    labels_pred = clusterer.predict(X)

    # inertia.append(algorithm.inertia_)
    ari = metrics.adjusted_rand_score(Y, labels_pred)
    print("n_clusters, ", n)
    ARI.append(ari)


np.save("data/me_PCA_ARI.npy",ARI)


ARI_EM = np.load('data/me_PCA_ARI.npy', allow_pickle=True)
ARI_ARR_EM = np.array(ARI_ARR)
ARI_kmeans = np.load('data/kmeans_PSA_ARI.npy', allow_pickle=True)
ARI_ARR_kmeans = np.array(ARI_ARR)
#print(ARI_ARR, ARI)
max_value_em = 11
max_value_kmeans = 15

fig, ax = plt.subplots()
plt.plot(ARI_ARR_EM, ARI_EM , 'o', color = 'steelblue', markersize = 2)
plt.plot(ARI_ARR_EM, ARI_EM , '-', color = 'steelblue' , alpha = 0.5, label ='EM')
plt.plot(ARI_ARR_EM[max_value_em-2], ARI_EM[max_value_em-2], 'ro')

plt.plot(ARI_ARR_kmeans, ARI_kmeans , 'o', color = 'orange', markersize = 2)
plt.plot(ARI_ARR_kmeans, ARI_kmeans , '-', color = 'orange' , alpha = 0.5, label ='Kmeans')
plt.plot(ARI_ARR_kmeans[max_value_kmeans-2], ARI_kmeans[max_value_kmeans-2], 'ro')
plt.title("ARI, PCA (Expectation maximization and Kmeans). ")
plt.xlabel('Number of Clusters') , plt.ylabel('ARI')
ax.legend(loc='best', frameon=True)
plt.savefig("plots/mnist_PCA_ARI.png")

