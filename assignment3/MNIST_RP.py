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


from sklearn import random_projection
from sklearn.decomposition import TruncatedSVD
from sklearn.random_projection import sparse_random_matrix
from sklearn import decomposition, random_projection


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
print("RP analysis")
"""
N_COMPS = np.arange(2,400,10)
REC_ARR = []
for i in range(0,10):
    print(i, "iteration")
    REC_ERROR = []
    for n_components in N_COMPS:
        print("n_components", n_components)
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

fig, ax = plt.subplots()
plt.plot(N_COMPS , rec_mean , 'o', color = 'steelblue', markersize = 2 )
plt.plot(N_COMPS, rec_mean, '-', color = 'steelblue' , alpha = 0.5)
plt.fill_between(N_COMPS, rec_mean - rec_std,
                 rec_mean + rec_std, alpha=0.2,
                 color="steelblue")

plt.title("Reconstruction error, Randomized projections.")
plt.xlabel('Number of components') , plt.ylabel('Reconstruction error')
plt.savefig("plots/MNIST_rp_rrecerror.png",bbox_inches='tight')

X = np.copy(X_safe)
Y = np.copy(Y_safe)
scaler = StandardScaler()
X = scaler.fit_transform(X)
transformer = InvertibleRandomProjection(n_components=350, random_state=np.random.randint(10000))
X_new = transformer.fit_transform(X)



range_n_clusters =np.arange(2,20,1)
#range_n_clusters = np.arange(2,3,1)
SIL_ARR =[]

for i in range(0,10):
    SILHOUETTES = []
    for n_clusters in range_n_clusters:
        # Create a subplot with 1 row and 2 columns
        X = np.copy(X_safe)
        Y = np.copy(Y_safe)
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        transformer = InvertibleRandomProjection(n_components=350, random_state = np.random.randint(10000))
        X = transformer.fit_transform(X)
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
    SIL_ARR.append(SILHOUETTES)
SIL_ARR = np.array(SIL_ARR)


np.save("data/mnist_kmeans_RP_sils.npy",SIL_ARR)

SIL_ARR = np.load('data/mnist_kmeans_RP_sils.npy', allow_pickle=True)
sil_mean = np.mean(SIL_ARR, axis = 0)
sil_std  = np.std(SIL_ARR,  axis = 0)


fig, ax = plt.subplots()
plt.plot(range_n_clusters , sil_mean , 'o', color = 'steelblue')
plt.plot(range_n_clusters, sil_mean  , '-', color = 'steelblue', alpha = 0.5)
plt.fill_between(range_n_clusters, sil_mean - sil_std,
                 sil_mean + sil_std, alpha=0.2,
                 color="steelblue")
# !!!! fix the optimal value
#plt.plot(range_n_clusters[max_value-2], SILHOUETTES[max_value-2], 'ro')
plt.title("Sillhouettes score vs number of clusters  (KMeans), Randomized projections")
plt.xlabel('Number of Clusters') , plt.ylabel('Sillhouettes score')
plt.savefig("plots/mnist_RP_kmeans_sillhouttes.png")


range_n_clusters =np.arange(2,20,1)
#range_n_clusters = np.arange(2,3,1)
SIL_ARR =[]
for i in range(0,10):
    SILHOUETTES = []
    for n_clusters in range_n_clusters:
        # Create a subplot with 1 row and 2 columns
        X = np.copy(X_safe)
        Y = np.copy(Y_safe)
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        transformer = InvertibleRandomProjection(n_components=350, random_state = np.random.randint(10000))
        X = transformer.fit_transform(X)
        # Initialize the clusterer with n_clusters value and a random generator
        # seed of 10 for reproducibility.
        clusterer = GaussianMixture(n_components=n_clusters, random_state=11, covariance_type='full')
        clusterer.fit(X)
        cluster_labels = clusterer.predict(X)

        # The silhouette_score gives the average value for all the samples.
        # This gives a perspective into the density and separation of the formed
        # clusters
        silhouette_avg = silhouette_score(X, cluster_labels)
        print("For n_clusters =", n_clusters,
          "The average silhouette_score is :", silhouette_avg)
        SILHOUETTES.append(silhouette_avg)
    SIL_ARR.append(SILHOUETTES)
SIL_ARR = np.array(SIL_ARR)


np.save("data/mnist_em_RP_sils.npy",SIL_ARR)
SIL_ARR = np.load('data/mnist_em_RP_sils.npy', allow_pickle=True)
sil_mean = np.mean(SIL_ARR, axis = 0)
sil_std  = np.std(SIL_ARR,  axis = 0)

fig, ax = plt.subplots()
plt.plot(range_n_clusters , sil_mean , 'o', color = 'steelblue')
plt.plot(range_n_clusters, sil_mean , '-', color = 'steelblue', alpha = 0.5)
plt.fill_between(range_n_clusters, sil_mean - sil_std,
                 sil_mean + sil_std, alpha=0.2,
                 color="steelblue")
# !!!! fix the optimal value
#plt.plot(range_n_clusters[max_value-2], SILHOUETTES[max_value-2], 'ro')
plt.title("Sillhouettes score vs number of clusters  (EM), Randomized projections.")
plt.xlabel('Number of Clusters') , plt.ylabel('Sillhouettes score')
plt.savefig("plots/mnist_RP_em_sillhouttes.png")





"""
inertia           = []
ARI_ARR           = np.arange(2,20,1)
ARI_FULL_GAUSSIAN = []
ARI_FULL_KMEANS   = []
for i in range(0,10):
    ARI_GAUSSIAN = []
    ARI_KMEANS   = []
    for n in ARI_ARR:
        X = np.copy(X_safe)
        Y = np.copy(Y_safe)
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        transformer = InvertibleRandomProjection(n_components=350, random_state=np.random.randint(10000))
        X = transformer.fit_transform(X)
        clusterer_gaussian = GaussianMixture(n_components=n, random_state=111, covariance_type='full')
        clusterer_gaussian.fit(X)
        labels_pred = clusterer_gaussian.predict(X)

        ari_gaussian = metrics.adjusted_rand_score(Y, labels_pred)
        print("n_clusters, ", n)
        algorithm = KMeans(n_clusters=n, init='k-means++', n_init=30, max_iter=1000,
                           tol=0.0001, random_state=111, algorithm='elkan')

        algorithm.fit(X)
        inertia.append(algorithm.inertia_)
        labels_pred = algorithm.labels_
        print("n_clusters, ", n)
        ARI_KMEANS.append(metrics.adjusted_rand_score(Y, labels_pred))
        ARI_GAUSSIAN.append(ari_gaussian)
    ARI_FULL_GAUSSIAN.append(ARI_GAUSSIAN)
    ARI_FULL_KMEANS.append(ARI_KMEANS)
ARI_FULL_GAUSSIAN = np.array(ARI_FULL_GAUSSIAN)
ARI_FULL_KMEANS   = np.array(ARI_FULL_KMEANS)
print(ARI_FULL_GAUSSIAN.shape)
print(len(ARI_ARR))
ari_gaussian_mean = np.mean(ARI_FULL_GAUSSIAN, axis = 0)
ari_gaussian_std = np.std(ARI_FULL_GAUSSIAN, axis = 0)
ari_kmeans_mean = np.mean(ARI_FULL_KMEANS, axis = 0)
ari_kmeans_std = np.std(ARI_FULL_KMEANS, axis = 0)
print(ari_gaussian_std)
print(ari_kmeans_std)

#print(ARI_ARR, ARI)

fig, ax = plt.subplots()
plt.plot(ARI_ARR, ari_gaussian_mean , 'o', color = 'steelblue', markersize = 2)
plt.plot(ARI_ARR, ari_gaussian_mean , '-', color = 'steelblue' , alpha = 0.5, label ='EM')
plt.fill_between(ARI_ARR, ari_gaussian_mean - ari_gaussian_std,
                 ari_gaussian_mean + ari_gaussian_std, alpha=0.2,
                 color="steelblue")

plt.plot(ARI_ARR, ari_kmeans_mean , 'o', color = 'orange', markersize = 2)
plt.plot(ARI_ARR, ari_kmeans_mean , '-', color = 'orange' , alpha = 0.5, label ='Kmeans')
plt.fill_between(ARI_ARR, ari_kmeans_mean - ari_kmeans_std,
                 ari_kmeans_mean + ari_kmeans_std, alpha=0.2,
                 color="orange")

plt.title("ARI, ICA (Expectation maximization and Kmeans). ")
plt.xlabel('Number of Clusters') , plt.ylabel('ARI')
ax.legend(loc='best', frameon=True)
plt.savefig("plots/mnist_RP_ARI.png")
