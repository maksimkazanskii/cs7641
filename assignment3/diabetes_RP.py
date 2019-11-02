# https://stackoverflow.com/questions/32750915/pca-inverse-transform-manually
# https://scprep.readthedocs.io/en/stable/_modules/scprep/reduce.html

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
from sklearn import decomposition, random_projection

def get_data_diabetes():
    data = pd.read_csv("data/diabetes.csv")
    data_np = data.as_matrix()

    Y = data_np[:, -1]
    Y = Y.astype('int')
    X = data_np[:, :-1]
    return X,Y
"""
def my_inverse_transform(transformer, X_tranformed):

def calculate_rec_error(X_transformed, X_original, transformer):
    X_train_pca_diff = (X_original - transformer!!!!.mean_).dot(transformer.components_.T)
    X_projected = my_inverse_transform(transformer, X_transformed)

    loss = ((X_original - X_projected) ** 2).mean()
"""


class InvertibleRandomProjection(random_projection.GaussianRandomProjection):
    """Gaussian random projection with an inverse transform using the pseudoinverse."""
    def __init__(
        self, n_components="auto", eps=0.3, orthogonalize=False, random_state=None
    ):
        self.orthogonalize = orthogonalize
        super().__init__(n_components=n_components, eps=eps, random_state=random_state)

    @property
    def pseudoinverse(self):
        """Pseudoinverse of the random projection

        This inverts the projection operation for any vector in the span of the
        random projection. For small enough `eps`, this should be close to the
        correct inverse.
        """
        try:
            return self._pseudoinverse
        except AttributeError:
            if self.orthogonalize:
                # orthogonal matrix: inverse is just its transpose
                self._pseudoinverse = self.components_
            else:
                self._pseudoinverse = np.linalg.pinv(self.components_.T)
            return self._pseudoinverse

    def fit(self, X):
        super().fit(X)
        if self.orthogonalize:
            Q, _ = np.linalg.qr(self.components_.T)
            self.components_ = Q.T
        return self


    def inverse_transform(self, X):
        return X.dot(self.pseudoinverse)

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
print("SVD analysis")
from sklearn.decomposition import FastICA
from scipy.stats import kurtosis
KURTOSIS = []
N_COMPS = np.arange(2,20,1)
from sklearn import random_projection
from sklearn.decomposition import TruncatedSVD
from sklearn.random_projection import GaussianRandomProjection

N_COMPS = np.arange(2,9,1)
REC_ARR = []
for i in range(0,10):
    REC_ERROR = []
    for n_components in N_COMPS:
        X = np.copy(X_safe)
        Y = np.copy(Y_safe)
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        transformer = InvertibleRandomProjection(n_components=n_components,  random_state=np.random.randint(10000))
        X_new = transformer.fit_transform(X)
        X_reconstructed = transformer.inverse_transform(X_new)
        loss = ((X - X_reconstructed) ** 2).mean()
        REC_ERROR.append(loss)
    REC_ARR.append(REC_ERROR)
REC_ARR = np.array(REC_ARR)
rec_mean = np.mean(REC_ARR, axis = 0)
rec_std  = np.std(REC_ARR, axis = 0)
print(rec_mean, rec_std)
fig, ax = plt.subplots()
plt.plot(N_COMPS , rec_mean , 'o', color = 'steelblue', markersize = 2 )
plt.plot(N_COMPS, rec_mean, '-', color = 'steelblue' , alpha = 0.5)
plt.fill_between(N_COMPS, rec_mean - rec_std,
                 rec_mean + rec_std, alpha=0.2,
                 color="steelblue")

plt.title("Reconstruction error, Randomized projections.")
plt.xlabel('Number of components') , plt.ylabel('Reconstruction error')
plt.savefig("plots/diabetes_rp_rrecerror.png",bbox_inches='tight')







np.random.seed(111)
range_n_clusters = np.arange(2,8,1)

SILS = []
for i in range(0,10):
    SIL_ARR = []
    for n_clusters in range_n_clusters:
        X = np.copy(X_safe)
        Y = np.copy(Y_safe)
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        transformer = InvertibleRandomProjection(n_components= 6, random_state=np.random.randint(10000))
        X = transformer.fit_transform(X)
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
        SIL_ARR.append(silhouette_avg)
    SILS.append(SIL_ARR)

np.save("data/diabetes_rp_kmeans_sils.npy",SILS)
SILS = np.load('data/diabetes_rp_kmeans_sils.npy', allow_pickle=True)

SILHOUETTES = np.mean(SILS, axis = 0)
SILHOUETTES_STD = np.std(SILS, axis = 0)

fig, ax = plt.subplots()
plt.plot(range_n_clusters , SILHOUETTES , 'o', color = 'steelblue')
plt.plot(range_n_clusters, SILHOUETTES  , '-', color = 'steelblue', alpha = 0.5)
plt.fill_between(range_n_clusters, SILHOUETTES - SILHOUETTES_STD,
                 SILHOUETTES+SILHOUETTES_STD, alpha=0.2,
                 color="steelblue")
plt.title("Sillhouettes score vs number of clusters  (KMeans)")
plt.xlabel('Number of Clusters') , plt.ylabel('Sillhouettes score')
plt.savefig("plots/diabetes_rp_kmeans_sillhouttes.png",bbox_inches='tight')


np.random.seed(111)
range_n_clusters = np.arange(2,8,1)

SILS = []
for i in range(0,10):
    SIL_ARR = []
    for n_clusters in range_n_clusters:
        X = np.copy(X_safe)
        Y = np.copy(Y_safe)
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        transformer = InvertibleRandomProjection(n_components= 6, random_state=np.random.randint(10000))
        X = transformer.fit_transform(X)
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
        SIL_ARR.append(silhouette_avg)
    SILS.append(SIL_ARR)

np.save("data/diabetes_rp_kmeans_sils.npy",SILS)
SILS = np.load('data/diabetes_rp_kmeans_sils.npy', allow_pickle=True)

SILHOUETTES = np.mean(SILS, axis = 0)
SILHOUETTES_STD = np.std(SILS, axis = 0)

fig, ax = plt.subplots()
plt.plot(range_n_clusters , SILHOUETTES , 'o', color = 'steelblue')
plt.plot(range_n_clusters, SILHOUETTES  , '-', color = 'steelblue', alpha = 0.5)
plt.fill_between(range_n_clusters, SILHOUETTES - SILHOUETTES_STD,
                 SILHOUETTES+SILHOUETTES_STD, alpha=0.2,
                 color="steelblue")
plt.title("Sillhouettes score vs number of clusters  (EM)")
plt.xlabel('Number of Clusters') , plt.ylabel('Sillhouettes score')
plt.savefig("plots/diabetes_rp_em_sillhouttes.png",bbox_inches='tight')


ARI_ARR = np.arange(2,8,1)
ARI_FULL =[]
for i in range(0,10):
    ARI = []
    for n in ARI_ARR:
        X = np.copy(X_safe)
        Y = np.copy(Y_safe)
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        transformer = InvertibleRandomProjection(n_components= 6, random_state=np.random.randint(10000))
        X = transformer.fit_transform(X)
        # Create a subplot with 1 row and 2 columns
        # Initialize the clusterer with n_clusters value and a random generator
        # seed of 10 for reproducibility.
        clusterer = GaussianMixture(n_components=n, random_state=11, covariance_type='full')
        labels_pred = clusterer.fit_predict(X)
        print("n_clusters, ", n)
        ARI.append(metrics.adjusted_mutual_info_score(Y, labels_pred))

    ARI_FULL.append(ARI)
ari_mean =  np.mean(ARI_FULL, axis = 0)
ari_std  =  np.std(ARI_FULL,  axis = 0)
fig, ax = plt.subplots()
plt.plot(ARI_ARR, ari_mean , 'o', color = 'steelblue')
plt.plot(ARI_ARR, ari_mean , '-', color = 'steelblue' , alpha = 0.5)
plt.fill_between(ARI_ARR, ari_mean - ari_std,
                 ari_mean+ari_std, alpha=0.2,
                 color="steelblue")
plt.title("Mutual info score (EM algorithm). ")
plt.xlabel('Number of Clusters') , plt.ylabel('Mutual info score')
plt.savefig("plots/diabetes_rp_me_mutual.png",bbox_inches='tight')





ARI_ARR = np.arange(2,8,1)
ARI_FULL =[]
for i in range(0,10):
    ARI = []
    for n in ARI_ARR:
        X = np.copy(X_safe)
        Y = np.copy(Y_safe)
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        transformer = InvertibleRandomProjection(n_components= 6, random_state=np.random.randint(10000))
        X = transformer.fit_transform(X)

        clusterer = KMeans(n_clusters=n, random_state=1)
        cluster_labels = clusterer.fit_predict(X)
        print("n_clusters, ", n)
        ARI.append(metrics.adjusted_mutual_info_score(Y, cluster_labels))
    ARI_FULL.append(ARI)
ari_mean =  np.mean(ARI_FULL, axis = 0)
ari_std  =  np.std(ARI_FULL,  axis = 0)
fig, ax = plt.subplots()
plt.plot(ARI_ARR, ari_mean , 'o', color = 'steelblue')
plt.plot(ARI_ARR, ari_mean , '-', color = 'steelblue' , alpha = 0.5)
plt.fill_between(ARI_ARR, ari_mean - ari_std,
                 ari_mean+ari_std, alpha=0.2,
                 color="steelblue")
plt.title("Mutual info score (K-means algorithm). ")
plt.xlabel('Number of Clusters') , plt.ylabel('Mutual info score')
plt.savefig("plots/diabetes_rp_kmeans_mutual.png",bbox_inches='tight')

X = np.copy(X_safe)
Y = np.copy(Y_safe)
scaler = StandardScaler()
X = scaler.fit_transform(X)
transformer = InvertibleRandomProjection(n_components=6, random_state=np.random.randint(10000))
X = transformer.fit_transform(X)

n_clusters =2

clusterer = KMeans(n_clusters=n_clusters, random_state=1)
cluster_labels = clusterer.fit_predict(X)
silhouette_avg = silhouette_score(X, cluster_labels)
fig, ax1 = plt.subplots()


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
plt.savefig('plots/diabetes_pr_kmeans_2d',bbox_inches='tight')




n_clusters = 2

clusterer = GaussianMixture(n_components=n_clusters, random_state=11, covariance_type='full')
clusterer_labels = clusterer.fit_predict(X)
silhouette_avg = silhouette_score(X, cluster_labels)
fig, ax1 = plt.subplots()


fig, ax2 = plt.subplots()
# 2nd Plot showing the actual clusters formed
colors = cm.nipy_spectral(cluster_labels.astype(float) / n_clusters)
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

plt.suptitle(("Silhouette analysis for EM clustering on sample data "
              "with n_clusters = %d" % n_clusters),
        )
plt.savefig('plots/diabetes_pr_em_2d',bbox_inches='tight')