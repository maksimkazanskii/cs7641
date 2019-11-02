# from https://www.kaggle.com/endlesslethe/siwei-digit-recognizer-top20

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Reshape, Conv2D, AveragePooling2D, Flatten
from keras.layers import MaxPooling2D
from keras.optimizers import adam
from sklearn.metrics import accuracy_score
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import ShuffleSplit
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix
import data_prep as data
from tensorflow import set_random_seed
import time
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
import pprint
from numpy.random import seed
seed(1)
from tensorflow import set_random_seed
set_random_seed(2)

def int2float_grey(x):
    x = x / 255
    return x

# find the left egde
# Note: the problem is that I don't do the parrallel part
def find_left_edge(x):
    edge_left = []
    n_samples = x.shape[0]
    for k in range(n_samples):
        for j in range(size_img):
            for i in range(size_img):
                if (x[k, size_img*i+j] >= threshold_color):
                    edge_left.append(j)
                    break
            if (len(edge_left) > k):
                break
    return edge_left

# find the right egde
# Note: the problem is that I don't do the parrallel part
def find_right_edge(x):
    edge_right = []
    n_samples = x.shape[0]
    for k in range(n_samples):
        for j in range(size_img):
            for i in range(size_img):
                if (x[k, size_img*i+(size_img-1-j)] >= threshold_color):
                    edge_right.append(size_img-1-j)
                    break
            if (len(edge_right) > k):
                break
    return edge_right



# find the top egde
# Note: the problem is that I don't do the parrallel part
def find_top_edge(x):
    edge_top = []
    n_samples = x.shape[0]
    for k in range(n_samples):
        for i in range(size_img):
            for j in range(size_img):
                if (x[k, size_img*i+j] >= threshold_color):
                    edge_top.append(i)
                    break
            if (len(edge_top) > k):
                break
    return edge_top


# find the bottom egde
# Note: the problem is that I don't do the parrallel part
def find_bottom_edge(x):
    edge_bottom = []
    n_samples = x.shape[0]
    for k in range(n_samples):
        for i in range(size_img):
            for j in range(size_img):
                if (x[k, size_img*(size_img-1-i)+j] >= threshold_color):
                    edge_bottom.append(size_img-1-i)
                    break
            if (len(edge_bottom) > k):
                break
    return edge_bottom


# Note：when we do the stretch part by ourselves,there may be some blank cells
# when the scale factor is more than 2

from skimage import transform


def stretch_image(x):
    # get edges
    edge_left = find_left_edge(x)
    edge_right = find_right_edge(x)
    edge_top = find_top_edge(x)
    edge_bottom = find_bottom_edge(x)

    # cropping and resize
    n_samples = x.shape[0]
    x = x.reshape(n_samples, size_img, size_img)
    for i in range(n_samples):
        x[i] = transform.resize(x[i][edge_top[i]:edge_bottom[i] + 1, edge_left[i]:edge_right[i] + 1],
                                (size_img, size_img))
    x = x.reshape(n_samples, size_img ** 2)
    show_img(x)



from sklearn.decomposition import PCA
def get_pca(x_train, x_test):
    pca = PCA(n_components=0.95)
    pca.fit(x_train)
    x_train = pca.transform(x_train)
    x_test = pca.transform(x_test)
    print(x_train.shape, x_test.shape)
    return x_train, x_test

#https://stackoverflow.com/questions/29831489/convert-array-of-indices-to-1-hot-encoded-numpy-array
def to_onehot(a):
    max_value = np.amax(a)
    b = np.zeros(( a.shape[0], max_value+1))
    b[np.arange(a.shape[0]), a] = 1
    return b

def to_onehot_empty(a,n_clusters):
    max_value = np.amax(a)
    b = np.zeros(( a.shape[0], n_clusters))
    b[np.arange(a.shape[0]), a] = 1
    return b

def general_function(mod_name, model_name):
    y_pred = model_train_predict(mod_name, model_name)
    output_prediction(y_pred, model_name)


from sklearn.model_selection import cross_val_score
def model_train_predict(mod_name, model_name):
    import_mod = __import__(mod_name, fromlist = str(True))
    if hasattr(import_mod, model_name):
         f = getattr(import_mod, model_name)
    else:
        print("404")
        return []
    clf = f()
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_train)
    get_acc(y_pred, y_train)
    scores = cross_val_score(clf, x_train, y_train, cv=5)
    print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
    y_pred = clf.predict(x_test)
    return y_pred

def get_acc(y_pred, y_train):
    right_num = (y_train == y_pred).sum()
    print("acc: ", right_num/n_samples_train)


def output_prediction(y_pred, model_name):
    print(y_pred)
    data_predict = {"ImageId":range(1, n_samples_test+1), "Label":y_pred}
    data_predict = pd.DataFrame(data_predict)
    data_predict.to_csv("dr output %s.csv" %model_name, index = False)

def score(y_true, y_pred, type):

    if type=="accuracy":
        #pprint.pprint(y_true)
        #pprint.pprint(y_pred)
        return accuracy_score(y_true, y_pred)

def from_onehot(g):

    vector = np.where(g == 1)[1]
    return vector

def all_score(y_true, y_pred):

    precision = precision_score(y_true, y_pred, average='macro')
    recall = recall_score(y_true, y_pred, average = 'macro')
    accuracy = accuracy_score(y_true, y_pred)
    return (accuracy, precision, recall)



dataset_size = 6000
size_img = 28
threshold_color = 100 / 255

file = open("data/mnist_train.csv")
data= pd.read_csv(file)

y = np.array(data.iloc[:, 0])
x = np.array(data.iloc[:, 1:])

random_indexes =np.random.permutation(x.shape[0])[:dataset_size]
x = x[random_indexes]
y = y[random_indexes]

x = preprocessing.normalize(x, norm='l2', axis=1)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=11)
x_train_cv, x_test_cv, y_train_cv, y_test_cv = train_test_split(x_train,y_train, test_size = 0.2, random_state =11)

n_features_train = x_train_cv.shape[1]
n_samples_train = x_train_cv.shape[0]
n_features_test = x_test_cv.shape[1]
n_samples_test = x_test_cv.shape[0]
print(n_features_train, n_samples_train, n_features_test, n_samples_test)
print(x_train_cv.shape, y_train_cv.shape, x_test_cv.shape, y_test_cv.shape)

#
# Preproccesing
#

x_train_cv = int2float_grey(x_train_cv)
x_test_cv = int2float_grey(x_test_cv)
x_train = int2float_grey(x_train)
x_test = int2float_grey(x_test)

y_train_cv = keras.utils.to_categorical(y_train_cv, num_classes=10)
y_test_cv = keras.utils.to_categorical(y_test_cv, num_classes=10)
y_train   = keras.utils.to_categorical(y_train, num_classes=10)
y_test   = keras.utils.to_categorical(y_test, num_classes=10)

y_all_pred = np.zeros((3, n_samples_test)).astype(np.int64)
print(y_all_pred.dtype)

# Basic performance
np.random.seed(111)
time0=time.time()

n_nodes = 11
model_name = "CNN"
model = Sequential()
model.add(Reshape(target_shape=(1, 28, 28), input_shape=(784,)))
model.add(Flatten())
model.add(Dense(output_dim=64, activation='relu'))
model.add(Dense(output_dim=10, activation='softmax'))
adam = keras.optimizers.Adam(lr=0.01, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])

model.fit(x_train_cv, y_train_cv, epochs=150, batch_size=128)

y_pred_train_cv = model.predict_classes(x_train_cv)
y_pred_test_cv = model.predict_classes(x_test_cv)

accuracy_train_cv = score(from_onehot(y_train_cv), y_pred_train_cv, "accuracy")
accuracy_test_cv  = score (from_onehot(y_test_cv), y_pred_test_cv, "accuracy")
benchmark_result = (accuracy_train_cv, accuracy_test_cv)
np.save("data/NN_benchmark_result.npy",benchmark_result)

(accuracy_train_cv, accuracy_test_cv) = np.load('data/NN_benchmark_result.npy', allow_pickle=True)
time1 = time.time()
print("REGULAR FEATURES", "accuracy_train_cv: ", accuracy_train_cv, "accuracy_test_cv: ", accuracy_test_cv, "Time ", time1-time0)
"""


Kmeans clustering


"""

def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

np.random.seed(111)
range_n_clusters = np.arange(2,50,1)
SILHOUETTES = []
TRAIN_ACCURACY = []
TEST_ACCURACY  = []
TRAIN_TIME     = []

for n_clusters in range_n_clusters:
    time0 = time.time()
    clusterer = KMeans(n_clusters=n_clusters, random_state=1)
    clusterer.fit(x_train_cv)
    train_labels_dirty = clusterer.predict(x_train_cv)
    train_labels = to_onehot(train_labels_dirty)

    val_labels_dirty   =  clusterer.predict(x_test_cv)
    val_labels = to_onehot(val_labels_dirty)

    n_nodes = 11
    model_name = "CNN"
    model = Sequential()
    model.add(Reshape(target_shape=(1, 1, n_clusters), input_shape=(n_clusters,)))
    model.add(Flatten())
    model.add(Dense(output_dim=64, activation='relu'))
    model.add(Dense(output_dim=10, activation='softmax'))
    adam = keras.optimizers.Adam(lr=0.01, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])

    model.fit(train_labels, y_train_cv, epochs=150, batch_size=128, verbose = False)

    y_pred_train_cv = model.predict_classes(train_labels)
    y_pred_test_cv = model.predict_classes(val_labels)

    accuracy_train_cv = score(from_onehot(y_train_cv), y_pred_train_cv, "accuracy")
    accuracy_test_cv = score(from_onehot(y_test_cv), y_pred_test_cv, "accuracy")
    TRAIN_ACCURACY.append(accuracy_train_cv)
    TEST_ACCURACY.append(accuracy_test_cv)
    time1 = time.time()
    TRAIN_TIME.append(time1-time0)
    print("N_clusters", n_clusters,"Training, validation accuracy: ",accuracy_train_cv, accuracy_test_cv)

TRAIN_TIME = np.array(TRAIN_TIME)
np.save("data/NN_kmeans_train.npy",TRAIN_ACCURACY)
np.save("data/NN_kmeans_test.npy",TEST_ACCURACY)
np.save("data/NN_kmeans_time.npy", TRAIN_TIME)

TRAIN_ACCURACY = np.load('data/NN_kmeans_train.npy', allow_pickle=True)
TEST_ACCURACY  = np.load('data/NN_kmeans_test.npy', allow_pickle=True)
TRAIN_TIME     = np.load('data/NN_kmeans_time.npy', allow_pickle=True)


fig, ax = plt.subplots()
X = range_n_clusters
ax.plot(X, TRAIN_ACCURACY, color='steelblue', label=" Training accuracy.", linewidth=2.0)
ax.plot(X, TRAIN_ACCURACY, color='steelblue', marker='o', markersize=4)

ax.plot(X, TEST_ACCURACY, color='red', label="Validation accuracy.", linewidth=2.0)
ax.plot(X, TEST_ACCURACY, color='red', marker='o', markersize=4)

ax.legend(loc='best loc', frameon=True)
plt.grid(True, linestyle='--')
plt.title("Validation curve (NN) for Kmeans clustering")
plt.xlabel(" Number of clusters")
plt.ylabel(" Accuracy ")
plt.savefig('plots/NN_clustering_kmeans.png')



fig, ax = plt.subplots()
X = range_n_clusters
ax.plot(X, TRAIN_TIME, color='steelblue', label=" Training accuracy.", linewidth=2.0)
ax.plot(X, TRAIN_TIME, color='steelblue', marker='o', markersize=4)

ax.legend(loc='best loc', frameon=True)
plt.grid(True, linestyle='--')
plt.title("Computational time")
plt.xlabel(" Number of clusters")
plt.ylabel(" Time (seconds)")
plt.savefig('plots/NN_clustering_time_kmeans.png')




# Full train and test
time0 =time.time()
n_clusters = 23

clusterer = KMeans(n_clusters=n_clusters, random_state=1)
clusterer.fit(x_train)
train_labels_dirty = clusterer.predict(x_train)
train_labels = to_onehot(train_labels_dirty)
val_labels_dirty = clusterer.predict(x_test)
val_labels = to_onehot(val_labels_dirty)

n_nodes = 11
model_name = "CNN"
model = Sequential()
model.add(Reshape(target_shape=(1, 1, n_clusters), input_shape=(n_clusters,)))
model.add(Flatten())
model.add(Dense(output_dim=64, activation='relu'))
model.add(Dense(output_dim=10, activation='softmax'))
adam = keras.optimizers.Adam(lr=0.01, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])

model.fit(train_labels, y_train, epochs=5, batch_size=128, verbose=False)

y_pred_train_cv = model.predict_classes(train_labels)
y_pred_test_cv = model.predict_classes(val_labels)

accuracy_train_cv = score(from_onehot(y_train), y_pred_train_cv, "accuracy")
accuracy_test_cv = score(from_onehot(y_test), y_pred_test_cv, "accuracy")
time1 = time.time()
print("N_clusters", n_clusters, " Full Training, testing accuracy (K means) ,time: ", accuracy_train_cv, accuracy_test_cv, time1-time0)




"""
Gaussian Mixture
"""

def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

np.random.seed(111)
range_n_clusters = np.arange(2,50,1)
SILHOUETTES = []
TRAIN_ACCURACY = []
TEST_ACCURACY  = []
TRAIN_TIME     = []

for n_clusters in range_n_clusters:
    time0 = time.time()
    clusterer =  GaussianMixture(n_components=n_clusters, random_state=1, init_params ='random',
                                 covariance_type='diag',  reg_covar = 1e-8)

    clusterer.fit(x_train_cv)
    train_labels_dirty = clusterer.predict(x_train_cv)
    train_labels = to_onehot_empty(train_labels_dirty, n_clusters)

    val_labels_dirty   =  clusterer.predict(x_test_cv)
    val_labels = to_onehot_empty(val_labels_dirty, n_clusters)



    n_nodes = 11
    model_name = "CNN"
    model = Sequential()
    model.add(Reshape(target_shape=(1, 1, n_clusters), input_shape=(n_clusters,)))
    model.add(Flatten())
    model.add(Dense(output_dim=64, activation='relu'))
    model.add(Dense(output_dim=10, activation='softmax'))
    adam = keras.optimizers.Adam(lr=0.01, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])

    model.fit(train_labels, y_train_cv, epochs=150, batch_size=128, verbose = False)

    y_pred_train_cv = model.predict_classes(train_labels)
    y_pred_test_cv = model.predict_classes(val_labels)

    accuracy_train_cv = score(from_onehot(y_train_cv), y_pred_train_cv, "accuracy")
    accuracy_test_cv = score(from_onehot(y_test_cv), y_pred_test_cv, "accuracy")
    TRAIN_ACCURACY.append(accuracy_train_cv)
    TEST_ACCURACY.append(accuracy_test_cv)
    time1 = time.time()
    TRAIN_TIME.append(time1-time0)
    print("N_clusters", n_clusters,"Training, validation accuracy: ",accuracy_train_cv, accuracy_test_cv)

TRAIN_TIME = np.array(TRAIN_TIME)
np.save("data/NN_gmm_train.npy",TRAIN_ACCURACY)
np.save("data/NN_gmm_test.npy",TEST_ACCURACY)
np.save("data/NN_gmm_time.npy", TRAIN_TIME)

TRAIN_ACCURACY = np.load('data/NN_gmm_train.npy', allow_pickle=True)
TEST_ACCURACY  = np.load('data/NN_gmm_test.npy', allow_pickle=True)
TRAIN_TIME     = np.load('data/NN_gmm_time.npy', allow_pickle=True)


fig, ax = plt.subplots()
X = range_n_clusters
#ax.plot(X, TRAIN_ACCURACY, color='steelblue', label=" Training accuracy.", linewidth=2.0)
#ax.plot(X, TRAIN_ACCURACY, color='steelblue', marker='o', markersize=4)

ax.plot(X, TEST_ACCURACY, color='red', label="Validation accuracy.", linewidth=2.0)
ax.plot(X, TEST_ACCURACY, color='red', marker='o', markersize=4)

ax.legend(loc='best loc', frameon=True)
plt.grid(True, linestyle='--')
plt.title("Validation curve (NN) for Expectation Maximization=")
plt.xlabel(" Number of clusters")
plt.ylabel(" Accuracy ")
plt.savefig('plots/NN_clustering_gmm.png')



fig, ax = plt.subplots()
X = range_n_clusters
ax.plot(X, TRAIN_TIME, color='steelblue', label=" Training accuracy.", linewidth=2.0)
ax.plot(X, TRAIN_TIME, color='steelblue', marker='o', markersize=4)

ax.legend(loc='best loc', frameon=True)
plt.grid(True, linestyle='--')
plt.title("Computational time (Expectation Maximization)")
plt.xlabel(" Number of clusters")
plt.ylabel(" Time (seconds)")
plt.savefig('plots/NN_clustering_time_gmm.png')




# Full train and test
time0 = time.time()
n_clusters = 27

clusterer = GaussianMixture(n_components=n_clusters, random_state=1, init_params='random',
                            covariance_type='diag', reg_covar=1e-8)

clusterer.fit(x_train)
train_labels_dirty = clusterer.predict(x_train)
train_labels = to_onehot_empty(train_labels_dirty, n_clusters)

val_labels_dirty = clusterer.predict(x_test)
val_labels = to_onehot_empty(val_labels_dirty, n_clusters)

n_nodes = 11
model_name = "CNN"
model = Sequential()
model.add(Reshape(target_shape=(1, 1, n_clusters), input_shape=(n_clusters,)))
model.add(Flatten())
model.add(Dense(output_dim=64, activation='relu'))
model.add(Dense(output_dim=10, activation='softmax'))
adam = keras.optimizers.Adam(lr=0.01, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])

model.fit(train_labels, y_train, epochs=150, batch_size=128, verbose=False)

y_pred_train_cv = model.predict_classes(train_labels)
y_pred_test_cv = model.predict_classes(val_labels)

accuracy_train_cv = score(from_onehot(y_train), y_pred_train_cv, "accuracy")
accuracy_test_cv = score(from_onehot(y_test), y_pred_test_cv, "accuracy")
time1 = time.time()
print("N_clusters (epxectiation maximization): ", n_clusters, "Training, testing accuracy, time: ", accuracy_train_cv, accuracy_test_cv, time1-time0)