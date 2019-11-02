## https://stackoverflow.com/questions/32750915/pca-inverse-transform-manually
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
from sklearn.mixture import GaussianMixture

X_full,Y_full = data.get_data_MNIST()
X_full = preprocessing.normalize(X_full, norm='l2', axis=1)
X, X_valtest, Y, Y_valtest = train_test_split(X_full, Y_full, test_size=0.4, random_state =111)

#X_full,Y_full = get_data_heart()
#print("X shape", X_full.shape)
#print("Y shape", Y_full.shape)


X_safe = np.copy(X)
Y_safe = np.copy(Y)

#
# SVD
#

X = np.copy(X_safe)
Y = np.copy(Y_safe)
print("SVD analysis")

from sklearn import random_projection
from sklearn.decomposition import TruncatedSVD
from sklearn.random_projection import sparse_random_matrix



X = np.copy(X_safe)
Y = np.copy(Y_safe)
scaler = StandardScaler()
X = scaler.fit_transform(X)
transformer = TruncatedSVD(n_components=100, n_iter=10, random_state=111)
X = transformer.fit_transform(X)
ratios      = transformer.explained_variance_ratio_

ratios = np.cumsum(ratios)
fig, ax = plt.subplots()
plt.plot(np.arange(1,len(ratios)+1,1) , ratios , 'o', color = 'steelblue', markersize = 2 )
plt.plot(np.arange(1,len(ratios)+1,1) , ratios , '-', color = 'steelblue' , alpha = 0.5)
plt.axhline(y=0.8, color='r', linestyle='--')
plt.axhline(y=0.6, color='r', linestyle='--')
plt.title("Cummulative variance, truncated SVD")
plt.xlabel('Component number') , plt.ylabel('Cummulative variance')
plt.savefig("plots/MNIST_svd_cum_variance.png")
# Take 3 components and project our dataset on them



X = np.copy(X_safe)
Y = np.copy(Y_safe)
scaler = StandardScaler()
X = scaler.fit_transform(X)
transformer = TruncatedSVD(n_components=50, n_iter=10, random_state=111)
X = transformer.fit_transform(X)


SILHOUETTES = []
range_n_clusters =np.arange(2,20,1)
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

np.save("data/kmeans_SVG_sils.npy",SILHOUETTES)
SILHOUETTES = np.load('data/kmeans_SVG_sils.npy', allow_pickle=True)


fig, ax = plt.subplots()
plt.plot(range_n_clusters , SILHOUETTES , 'o', color = 'steelblue')
plt.plot(range_n_clusters, SILHOUETTES  , '-', color = 'steelblue', alpha = 0.5)
# !!!! fix the optimal value
#plt.plot(range_n_clusters[max_value-2], SILHOUETTES[max_value-2], 'ro')
plt.title("Sillhouettes score vs number of clusters  (KMeans), truncated SVG")
plt.xlabel('Number of Clusters') , plt.ylabel('Sillhouettes score')
plt.savefig("plots/mnist_svg_kmeans_sillhouttes.png")



np.random.seed(111)
range_n_clusters = np.arange(2,20,1)


SILHOUETTES = []

for n_clusters in range_n_clusters:
    # Create a subplot with 1 row and 2 columns

    # Initialize the clusterer with n_clusters value and a random generator
    # seed of 10 for reproducibility.

    clusterer = GaussianMixture(n_components=n_clusters, random_state=11, covariance_type='full')

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

np.save("data/me_svg_sils.npy",SILHOUETTES)
SILHOUETTES = np.load('data/me_svg_sils.npy', allow_pickle=True)


fig, ax = plt.subplots()
plt.plot(range_n_clusters , SILHOUETTES , 'o', color = 'steelblue')
plt.plot(range_n_clusters, SILHOUETTES  , '-', color = 'steelblue', alpha = 0.5)
# !!!! fix the optimal value
#plt.plot(range_n_clusters[max_value-2], SILHOUETTES[max_value-2], 'ro')
plt.title("Sillhouettes score vs number of clusters \n(EM), truncated SVG")
plt.xlabel('Number of Clusters') , plt.ylabel('Sillhouettes score')
plt.savefig("plots/mnist_svg_me_sils.png")





inertia = []
ARI = []
ARI_ARR = np.arange(2,20,1)

for n in ARI_ARR:

    clusterer = GaussianMixture(n_components=n, random_state=111, covariance_type='full')
    clusterer.fit(X)
    labels_pred = clusterer.predict(X)

    # inertia.append(algorithm.inertia_)
    ari = metrics.adjusted_rand_score(Y, labels_pred)
    print("n_clusters, ", n)
    ARI.append(ari)


np.save("data/me_SVG_ARI.npy",ARI)
ARI_EM = np.load('data/me_SVG_ARI.npy', allow_pickle=True)


inertia = []
ARI = []
ARI_ARR = np.arange(2,20,1)
for n in ARI_ARR:
    algorithm = KMeans(n_clusters = n ,init='k-means++', n_init = 30 ,max_iter=1000,
                        tol=0.0001,  random_state= 111  , algorithm='elkan')
    algorithm.fit(X)
    inertia.append(algorithm.inertia_)
    labels_pred = algorithm.labels_
    print("n_clusters, ", n)
    ARI.append(metrics.adjusted_rand_score(Y, labels_pred))
np.save("data/kmeans_SVG_ARI.npy",ARI)
ARI_kmeans = np.load('data/kmeans_SVG_ARI.npy', allow_pickle=True)


#print(ARI_ARR, ARI)


fig, ax = plt.subplots()
plt.plot(ARI_ARR, ARI_EM , 'o', color = 'steelblue', markersize = 2)
plt.plot(ARI_ARR, ARI_EM , '-', color = 'steelblue' , alpha = 0.5, label ='EM')

plt.plot(ARI_ARR, ARI_kmeans , 'o', color = 'orange', markersize = 2)
plt.plot(ARI_ARR, ARI_kmeans , '-', color = 'orange' , alpha = 0.5, label ='Kmeans')

plt.title("ARI, ICA (Expectation maximization and Kmeans). ")
plt.xlabel('Number of Clusters') , plt.ylabel('ARI')
ax.legend(loc='best', frameon=True)
plt.savefig("plots/mnist_SVG_ARI.png")


