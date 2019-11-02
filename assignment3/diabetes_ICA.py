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
from sklearn.mixture import GaussianMixture

def get_data_diabetes():
    data = pd.read_csv("data/diabetes.csv")
    data_np = data.as_matrix()

    Y = data_np[:, -1]
    Y = Y.astype('int')
    X = data_np[:, :-1]
    return X,Y


print("Diabetes start)")
np.random.seed(1111)
X,Y = get_data_diabetes()

print(X.shape)

X_safe = np.copy(X)
Y_safe = np.copy(Y)


X = np.copy(X_safe)
Y = np.copy(Y_safe)
scaler = StandardScaler()
X= scaler.fit_transform(X)
print("PCA analysis")
from sklearn.decomposition import FastICA
from scipy.stats import kurtosis
KURTOSIS = []
N_COMPS = np.arange(2,20,1)

for n_comps in N_COMPS:
    X = np.copy(X_safe)
    Y = np.copy(Y_safe)
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    transformer = FastICA(n_components=n_comps,  random_state=111,tol=0.001)
    X = transformer.fit_transform(X)
    kurt = kurtosis(X,axis=1)
    kurt = np.mean(np.abs(kurt))
    KURTOSIS.append(kurt)
    print(" n_clusters = ", n_comps, " Kurtosis :", kurtosis)

fig, ax = plt.subplots()
plt.plot(N_COMPS, KURTOSIS , 'o', color = 'steelblue')
plt.plot(N_COMPS, KURTOSIS , '-', color = 'steelblue' , alpha = 0.5)
plt.title("Kurtosis, Independent component analysis")
plt.xlabel('Number of components') , plt.ylabel('Average kurtosis')
plt.savefig("plots/diabetes_ICA_kurtosis.png")

X = np.copy(X_safe)
Y = np.copy(Y_safe)
scaler = StandardScaler()
X = scaler.fit_transform(X)
transformer = FastICA(n_components = 2, random_state=111, tol=0.001)
X = transformer.fit_transform(X)


np.random.seed(111)
range_n_clusters = np.arange(2,8,1)
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


np.save("data/diabetes_kmeans_ica_sils.npy",SILHOUETTES)
SILHOUETTES = np.load('data/diabetes_kmeans_ica_sils.npy', allow_pickle=True)

n_clusters = 3
clusterer = KMeans(n_clusters=n_clusters, random_state=1)
cluster_labels = clusterer.fit_predict(X)
silhouette_avg = silhouette_score(X, cluster_labels)
fig, ax1 = plt.subplots()

# The 1st subplot is the silhouette plot
# The silhouette coefficient can range from -1, 1 but in this example all
# lie within [-0.1, 1]
ax1.set_xlim([-0.1, 1])
# The (n_clusters+1)*10 is for inserting blank space between silhouette
# plots of individual clusters, to demarcate them clearly.
ax1.set_ylim([0, len(X) + (n_clusters + 1) * 10])

# Initialize the clusterer with n_clusters value and a random generator
# seed of 10 for reproducibility.
clusterer = KMeans(n_clusters=n_clusters, random_state=10)
cluster_labels = clusterer.fit_predict(X)

# The silhouette_score gives the average value for all the samples.
# This gives a perspective into the density and separation of the formed
# clusters
silhouette_avg = silhouette_score(X, cluster_labels)
print("For n_clusters =", n_clusters,
      "The average silhouette_score is :", silhouette_avg)

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
    ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i),fontsize='large')

    # Compute the new y_lower for next plot
    y_lower = y_upper + 10  # 10 for the 0 samples

ax1.set_title("The silhouette plot for the various clusters.",fontsize='large')
ax1.set_xlabel("The silhouette coefficient values",fontsize='large')
ax1.set_ylabel("Cluster label",fontsize='large')

# The vertical line for average silhouette score of all the values
ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

ax1.set_yticks([])  # Clear the yaxis labels / ticks
ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

plt.savefig('plots/diabetes_ica_clustering1',bbox_inches='tight')

fig, ax2 = plt.subplots()
# 2nd Plot showing the actual clusters formed
colors = cm.nipy_spectral(cluster_labels.astype(float) / n_clusters)
ax2.scatter(X[:, 0], X[:, 1], marker='.', s=30, lw=0, alpha=0.7,
            c=colors, edgecolor='k')

# Labeling the clusters
centers = clusterer.cluster_centers_
# Draw white circles at cluster centers
ax2.scatter(centers[:, 0], centers[:, 1], marker='o',
            c="white", alpha=1, s=200, edgecolor='k')

for i, c in enumerate(centers):
    ax2.scatter(c[0], c[1], marker='$%d$' % i, alpha=1,
                s=50, edgecolor='k')

ax2.set_title("The visualization of the clustered data.")
ax2.set_xlabel("Feature space for the 1st feature")
ax2.set_ylabel("Feature space for the 2nd feature")

plt.suptitle(("Silhouette analysis for KMeans clustering on sample data "
              "with n_clusters = %d" % n_clusters),
        )
plt.savefig('plots/diabetes_ica_clustering_2',bbox_inches='tight')




fig, ax = plt.subplots()
plt.plot(range_n_clusters , SILHOUETTES , 'o', color = 'steelblue')
plt.plot(range_n_clusters, SILHOUETTES  , '-', color = 'steelblue', alpha = 0.5)

plt.title("Sillhouettes score vs number of clusters  (KMeans)")
plt.xlabel('Number of Clusters') , plt.ylabel('Sillhouettes score')
plt.savefig("plots/diabetes_ica_kmeans_sillhouttes.png",bbox_inches='tight')

inertia = []
ARI = []
ARI_ARR = np.arange(2,15,1)

for n in ARI_ARR:
    algorithm = KMeans(n_clusters = n ,init='k-means++', n_init = 30 ,max_iter=1000,
                        tol=0.0001,  random_state= 111  , algorithm='elkan')

    algorithm.fit(X)
    inertia.append(algorithm.inertia_)
    labels_pred = algorithm.labels_
    print("n_clusters, ", n)
    ARI.append(metrics.adjusted_mutual_info_score(Y, labels_pred))

fig, ax = plt.subplots()
plt.plot(ARI_ARR, ARI , 'o', color = 'steelblue')
plt.plot(ARI_ARR, ARI , '-', color = 'steelblue' , alpha = 0.5)
plt.title("Mutual info score (K-Means algorithm). ")
plt.xlabel('Number of Clusters') , plt.ylabel('Mutual info score')
plt.savefig("plots/diabetes_ica_kmeans_mutual.png",bbox_inches='tight')


fig, ax = plt.subplots()
plt.plot(ARI_ARR, inertia , 'o', color = 'steelblue')
plt.plot(ARI_ARR, inertia , '-', color = 'steelblue' , alpha = 0.5)
plt.title("Inertia (K-Means algorithm). ",fontsize='large')
plt.xlabel('Number of Clusters',fontsize='large') , plt.ylabel('ARI',fontsize='large')
plt.savefig("plots/diabetes_ica_inertia.png",bbox_inches='tight')



# EM

np.random.seed(1111)
X,Y = get_data_diabetes()

print(X.shape)

X_safe = np.copy(X)
Y_safe = np.copy(Y)


X = np.copy(X_safe)
Y = np.copy(Y_safe)
scaler = StandardScaler()
X = scaler.fit_transform(X)
transformer = FastICA(n_components = 2, random_state=111, tol=0.001)
X = transformer.fit_transform(X)

np.random.seed(111)
range_n_clusters = np.arange(2,15,1)
SILHOUETTES = []

for n_clusters in range_n_clusters:
    # Create a subplot with 1 row and 2 columns

    # Initialize the clusterer with n_clusters value and a random generator
    # seed of 10 for reproducibility.
    clusterer = GaussianMixture(n_components=n_clusters, random_state=11, covariance_type='full')
    cluster_labels = clusterer.fit_predict(X)

    # The silhouette_score gives the average value for all the samples.
    # This gives a perspective into the density and separation of the formed
    # clusters
    silhouette_avg = silhouette_score(X, cluster_labels)
    print("For n_clusters =", n_clusters,
          "The average silhouette_score is :", silhouette_avg)
    SILHOUETTES.append(silhouette_avg)


np.save("data/diabetes_ica_me_sils.npy",SILHOUETTES)
SILHOUETTES = np.load('data/diabetes_ica_me_sils.npy', allow_pickle=True)


n_clusters = 2
clusterer = GaussianMixture(n_components=n_clusters, random_state=11, covariance_type='full')
cluster_labels = clusterer.fit_predict(X)
silhouette_avg = silhouette_score(X, cluster_labels)
fig, ax1 = plt.subplots()

# The 1st subplot is the silhouette plot
# The silhouette coefficient can range from -1, 1 but in this example all
# lie within [-0.1, 1]
ax1.set_xlim([-0.1, 1])
# The (n_clusters+1)*10 is for inserting blank space between silhouette
# plots of individual clusters, to demarcate them clearly.
ax1.set_ylim([0, len(X) + (n_clusters + 1) * 10])

# Initialize the clusterer with n_clusters value and a random generator
# seed of 10 for reproducibility.
clusterer = GaussianMixture(n_components=n_clusters, random_state=11, covariance_type='full')
cluster_labels = clusterer.fit_predict(X)

# The silhouette_score gives the average value for all the samples.
# This gives a perspective into the density and separation of the formed
# clusters

silhouette_avg = silhouette_score(X, cluster_labels)
print("For n_clusters =", n_clusters,
      "The average silhouette_score is :", silhouette_avg)

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
    ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i),fontsize='large')

    # Compute the new y_lower for next plot
    y_lower = y_upper + 10  # 10 for the 0 samples

ax1.set_title("The silhouette plot for the various clusters.",fontsize='large')
ax1.set_xlabel("The silhouette coefficient values",fontsize='large')
ax1.set_ylabel("Cluster label",fontsize='large')

# The vertical line for average silhouette score of all the values
ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

ax1.set_yticks([])  # Clear the yaxis labels / ticks
ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

plt.savefig('plots/diabetes_clustering1_me_ica.png',bbox_inches='tight')

fig, ax2 = plt.subplots()
# 2nd Plot showing the actual clusters formed
colors = cm.prism(cluster_labels.astype(float) / n_clusters)
ax2.scatter(X[:, 0], X[:, 1], marker='.', s=30, lw=0, alpha=0.7,
            c=colors, edgecolor='k')

# Labeling the clusters
centers = clusterer.means_
# Draw white circles at cluster centers
ax2.scatter(centers[:, 0], centers[:, 1], marker='o',
            c="white", alpha=1, s=200, edgecolor='k')

for i, c in enumerate(centers):
    ax2.scatter(c[0], c[1], marker='$%d$' % i, alpha=1,
                s=50, edgecolor='k')

ax2.set_title("The visualization of the clustered data.")
ax2.set_xlabel("Feature space for the 1st feature")
ax2.set_ylabel("Feature space for the 2nd feature")

plt.suptitle(("Silhouette analysis for KMeans clustering on sample data "
              "with n_clusters = %d" % n_clusters),
        )
plt.savefig('plots/diabetes_ica_clustering2_me.png',bbox_inches='tight')




fig, ax = plt.subplots()
plt.plot(range_n_clusters , SILHOUETTES , 'o', color = 'steelblue')
plt.plot(range_n_clusters, SILHOUETTES  , '-', color = 'steelblue', alpha = 0.5)

plt.title("Sillhouettes score vs number of clusters  (EM)")
plt.xlabel('Number of Clusters') , plt.ylabel('Sillhouettes score')
plt.savefig("plots/diabetes_ica_me_sillhouttes.png",bbox_inches='tight')


ARI = []
ARI_ARR = np.arange(2,15,1)

for n in ARI_ARR:
    algorithm =   GaussianMixture(n_components=n, random_state=11, covariance_type='full')

    algorithm.fit(X)
    algorithm.fit(X)
    labels_pred = algorithm.predict(X)
    print("n_clusters, ", n)
    ARI.append(metrics.adjusted_mutual_info_score(Y, labels_pred))

fig, ax = plt.subplots()
plt.plot(ARI_ARR, ARI , 'o', color = 'steelblue')
plt.plot(ARI_ARR, ARI , '-', color = 'steelblue' , alpha = 0.5)
plt.title("Mutual info score (EM algorithm). ")
plt.xlabel('Number of Clusters') , plt.ylabel('Mutual info score')
plt.savefig("plots/diabetes_ica_me_mutual.png",bbox_inches='tight')