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


def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

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


"""

PCA

"""

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

model.fit(x_train, y_train, epochs=150, batch_size=128)

y_pred_train_cv = model.predict_classes(x_train_cv)
y_pred_test_cv = model.predict_classes(x_test_cv)

accuracy_train_cv = score(from_onehot(y_train_cv), y_pred_train_cv, "accuracy")
accuracy_test_cv  = score (from_onehot(y_test_cv), y_pred_test_cv, "accuracy")
benchmark_result = (accuracy_train_cv, accuracy_test_cv)
np.save("data/NN_benchmark_result.npy",benchmark_result)

(accuracy_train_cv, accuracy_test_cv) = np.load('data/NN_benchmark_result.npy', allow_pickle=True)
time1 = time.time()
print("REGULAR FEATURES", "accuracy_train_cv: ", accuracy_train_cv, "accuracy_test_cv: ", accuracy_test_cv, "Time ", time1-time0)




print("\n\n\n\n\n PCA analysis")
from sklearn.decomposition import PCA
N_COMPS = np.arange(1,40,1)
TRAIN_ARR = []
TEST_ARR  = []
TIME_ARR  = []
"""
for n_comps in N_COMPS:
    pca = PCA(n_components=n_comps)
    x_train_transform = pca.fit_transform(x_train_cv)
    x_test_transform  = pca.transform(x_test_cv)

    time0 = time.time()

    n_nodes = 11
    model_name = "CNN"
    model = Sequential()
    model.add(Reshape(target_shape=(1, 1, n_comps), input_shape=(n_comps,)))
    model.add(Flatten())
    model.add(Dense(output_dim=64, activation='relu'))
    model.add(Dense(output_dim=10, activation='softmax'))
    adam = keras.optimizers.Adam(lr=0.01, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])

    model.fit(x_train_transform, y_train_cv, epochs=150, batch_size=128, verbose = False)

    y_pred_train_cv = model.predict_classes(x_train_transform)
    y_pred_test_cv = model.predict_classes(x_test_transform)

    accuracy_train_cv = score(from_onehot(y_train_cv), y_pred_train_cv, "accuracy")
    accuracy_test_cv = score(from_onehot(y_test_cv), y_pred_test_cv, "accuracy")


    time1 = time.time()
    TIME_ARR.append(time1-time0)
    TRAIN_ARR.append(accuracy_train_cv)
    TEST_ARR.append(accuracy_test_cv)
    print(n_comps, "accuracy_train_cv: ", accuracy_train_cv, "accuracy_test_cv: ", accuracy_test_cv, "Time ",
          time1 - time0)


np.save("data/NN_pca_train.npy",TRAIN_ARR)
np.save("data/NN_pca_test.npy",TEST_ARR)
np.save("data/NN_pca_time.npy", TIME_ARR)
"""
TRAIN_ACCURACY = np.load('data/NN_pca_train.npy', allow_pickle=True)
TEST_ACCURACY  = np.load('data/NN_pca_test.npy', allow_pickle=True)
TRAIN_TIME     = np.load('data/NN_pca_time.npy', allow_pickle=True)


fig, ax = plt.subplots()
X = N_COMPS
#ax.plot(X, TRAIN_ACCURACY, color='steelblue', label=" Training accuracy.", linewidth=2.0)
#ax.plot(X, TRAIN_ACCURACY, color='steelblue', marker='o', markersize=4)

ax.plot(X, TEST_ACCURACY, color='red', label="Validation accuracy.", linewidth=2.0)
ax.plot(X, TEST_ACCURACY, color='red', marker='o', markersize=4)

ax.legend(loc='best loc', frameon=True)
plt.grid(True, linestyle='--')
plt.title("Validation curve PCA")
plt.xlabel(" Number of components")
plt.ylabel(" Accuracy ")
plt.savefig('plots/NN_pca_acc.png',bbox_inches='tight')



fig, ax = plt.subplots()
X = N_COMPS
ax.plot(X, TRAIN_TIME, color='steelblue', label=" Training time", linewidth=2.0)
ax.plot(X, TRAIN_TIME, color='steelblue', marker='o', markersize=4)

ax.legend(loc='best loc', frameon=True)
plt.grid(True, linestyle='--')
plt.title("Computational time  (PCA)")
plt.xlabel(" Number of componenets")
plt.ylabel(" Time (seconds)")
plt.savefig('plots/NN_pca_time.png',bbox_inches='tight')


time0 = time.time()




pca = PCA(n_components=45)
x_train_transform = pca.fit_transform(x_train)
x_test_transform = pca.transform(x_test)

n_nodes = 11
n_comps = 45
model_name = "CNN"
model = Sequential()
model.add(Reshape(target_shape=(1, 1, n_comps), input_shape=(n_comps,)))
model.add(Flatten())
model.add(Dense(output_dim=64, activation='relu'))
model.add(Dense(output_dim=10, activation='softmax'))
adam = keras.optimizers.Adam(lr=0.01, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])

model.fit(x_train_transform, y_train, epochs=150, batch_size=128, verbose=False)

y_pred_train_cv = model.predict_classes(x_train_transform)
y_pred_test_cv = model.predict_classes(x_test_transform)

accuracy_train_cv = score(from_onehot(y_train), y_pred_train_cv, "accuracy")
accuracy_test_cv = score(from_onehot(y_test), y_pred_test_cv, "accuracy")

time1 = time.time()
TIME_ARR.append(time1 - time0)
TRAIN_ARR.append(accuracy_train_cv)
TEST_ARR.append(accuracy_test_cv)
print("PCA", "N_comps", n_comps, "accuracy_train_cv: ", accuracy_train_cv, "accuracy_test_cv: ", accuracy_test_cv, "Time ",
      time1 - time0)


"""

ICA

"""


print("\n\n\n\n\n ICA analysis")
from sklearn.decomposition import FastICA
N_COMPS = np.arange(1,40,1)
TRAIN_ARR = []
TEST_ARR  = []
TIME_ARR  = []
"""
for n_comps in N_COMPS:
    transformer = FastICA(n_components=n_comps, random_state=111, tol=0.001)

    x_train_transform = transformer.fit_transform(x_train_cv)
    x_test_transform  = transformer.transform(x_test_cv)

    time0 = time.time()

    n_nodes = 11
    model_name = "CNN"
    model = Sequential()
    model.add(Reshape(target_shape=(1, 1, n_comps), input_shape=(n_comps,)))
    model.add(Flatten())
    model.add(Dense(output_dim=64, activation='relu'))
    model.add(Dense(output_dim=10, activation='softmax'))
    adam = keras.optimizers.Adam(lr=0.01, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])

    model.fit(x_train_transform, y_train_cv, epochs=150, batch_size=128, verbose = False)

    y_pred_train_cv = model.predict_classes(x_train_transform)
    y_pred_test_cv = model.predict_classes(x_test_transform)

    accuracy_train_cv = score(from_onehot(y_train_cv), y_pred_train_cv, "accuracy")
    accuracy_test_cv = score(from_onehot(y_test_cv), y_pred_test_cv, "accuracy")


    time1 = time.time()
    TIME_ARR.append(time1-time0)
    TRAIN_ARR.append(accuracy_train_cv)
    TEST_ARR.append(accuracy_test_cv)
    print(n_comps, "accuracy_train_cv: ", accuracy_train_cv, "accuracy_test_cv: ", accuracy_test_cv, "Time ",
          time1 - time0)


np.save("data/NN_ica_train.npy",TRAIN_ARR)
np.save("data/NN_ica_test.npy",TEST_ARR)
np.save("data/NN_ica_time.npy", TIME_ARR)
"""
TRAIN_ACCURACY = np.load('data/NN_ica_train.npy', allow_pickle=True)
TEST_ACCURACY  = np.load('data/NN_ica_test.npy', allow_pickle=True)
TRAIN_TIME     = np.load('data/NN_ica_time.npy', allow_pickle=True)


fig, ax = plt.subplots()
X = N_COMPS
#ax.plot(X, TRAIN_ACCURACY, color='steelblue', label=" Training accuracy.", linewidth=2.0)
#ax.plot(X, TRAIN_ACCURACY, color='steelblue', marker='o', markersize=4)

ax.plot(X, TEST_ACCURACY, color='red', label="Validation accuracy.", linewidth=2.0)
ax.plot(X, TEST_ACCURACY, color='red', marker='o', markersize=4)

ax.legend(loc='best loc', frameon=True)
plt.grid(True, linestyle='--')
plt.title("Validation curve ICA")
plt.xlabel(" Number of components")
plt.ylabel(" Accuracy ")
plt.savefig('plots/NN_ica_acc.png', bbox_inches='tight')



fig, ax = plt.subplots()
X = N_COMPS
ax.plot(X, TRAIN_TIME, color='steelblue', label=" Training time", linewidth=2.0)
ax.plot(X, TRAIN_TIME, color='steelblue', marker='o', markersize=4)

ax.legend(loc='best loc', frameon=True)
plt.grid(True, linestyle='--')
plt.title("Computational time  (PCA)")
plt.xlabel(" Number of componenets")
plt.ylabel(" Time (seconds)")
plt.savefig('plots/NN_ica_time.png', bbox_inches='tight')



transformer = FastICA(n_components=602, random_state=111, tol=0.001)
x_train_transform = transformer.fit_transform(x_train)
x_test_transform  = transformer.transform(x_test)
n_nodes = 11
n_comps = 602
model_name = "CNN"
model = Sequential()
model.add(Reshape(target_shape=(1, 1, n_comps), input_shape=(n_comps,)))
model.add(Flatten())
model.add(Dense(output_dim=64, activation='relu'))
model.add(Dense(output_dim=10, activation='softmax'))
adam = keras.optimizers.Adam(lr=0.01, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])

model.fit(x_train_transform, y_train, epochs=150, batch_size=128, verbose=False)

y_pred_train_cv = model.predict_classes(x_train_transform)
y_pred_test_cv = model.predict_classes(x_test_transform)

accuracy_train_cv = score(from_onehot(y_train), y_pred_train_cv, "accuracy")
accuracy_test_cv = score(from_onehot(y_test), y_pred_test_cv, "accuracy")

time1 = time.time()
TIME_ARR.append(time1 - time0)
TRAIN_ARR.append(accuracy_train_cv)
TEST_ARR.append(accuracy_test_cv)
print("ICA N_COMPS: ", n_comps, "accuracy_train_cv: ", accuracy_train_cv, "accuracy_test_cv: ", accuracy_test_cv, "Time ",
      time1 - time0)

"""

Randomized projections

"""


print("\n\n\n\n Randomized projections")
from sklearn import random_projection
from sklearn.random_projection import SparseRandomProjection
N_COMPS = np.arange(1,40,1)
TRAIN_ARR = []
TEST_ARR  = []
TIME_ARR  = []
"""
for n_comps in N_COMPS:
    rng = np.random.RandomState(111)
    transformer = SparseRandomProjection(random_state=rng, n_components=n_comps)
    x_train_transform = transformer.fit_transform(x_train_cv)
    x_test_transform  = transformer.transform(x_test_cv)

    time0 = time.time()

    n_nodes = 11
    model_name = "CNN"
    model = Sequential()
    model.add(Reshape(target_shape=(1, 1, n_comps), input_shape=(n_comps,)))
    model.add(Flatten())
    model.add(Dense(output_dim=64, activation='relu'))
    model.add(Dense(output_dim=10, activation='softmax'))
    adam = keras.optimizers.Adam(lr=0.01, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])

    model.fit(x_train_transform, y_train_cv, epochs=150, batch_size=128, verbose = False)

    y_pred_train_cv = model.predict_classes(x_train_transform)
    y_pred_test_cv = model.predict_classes(x_test_transform)

    accuracy_train_cv = score(from_onehot(y_train_cv), y_pred_train_cv, "accuracy")
    accuracy_test_cv = score(from_onehot(y_test_cv), y_pred_test_cv, "accuracy")


    time1 = time.time()
    TIME_ARR.append(time1-time0)
    TRAIN_ARR.append(accuracy_train_cv)
    TEST_ARR.append(accuracy_test_cv)
    print(n_comps, "accuracy_train_cv: ", accuracy_train_cv, "accuracy_test_cv: ", accuracy_test_cv, "Time ",
          time1 - time0)


np.save("data/NN_RP_train.npy",TRAIN_ARR)
np.save("data/NN_RP_test.npy",TEST_ARR)
np.save("data/NN_RP_time.npy", TIME_ARR)
"""

TRAIN_ACCURACY = np.load('data/NN_RP_train.npy', allow_pickle=True)
TEST_ACCURACY  = np.load('data/NN_RP_test.npy', allow_pickle=True)
TRAIN_TIME     = np.load('data/NN_RP_time.npy', allow_pickle=True)


fig, ax = plt.subplots()
X = N_COMPS
#ax.plot(X, TRAIN_ACCURACY, color='steelblue', label=" Training accuracy.", linewidth=2.0)
#ax.plot(X, TRAIN_ACCURACY, color='steelblue', marker='o', markersize=4)

ax.plot(X, TEST_ACCURACY, color='red', label="Validation accuracy.", linewidth=2.0)
ax.plot(X, TEST_ACCURACY, color='red', marker='o', markersize=4)

ax.legend(loc='best loc', frameon=True)
plt.grid(True, linestyle='--')
plt.title("Validation curve, Randomized projections")
plt.xlabel(" Number of components")
plt.ylabel(" Accuracy ")
plt.savefig('plots/NN_RP_acc.png', bbox_inches='tight')



fig, ax = plt.subplots()
X = N_COMPS
ax.plot(X, TRAIN_TIME, color='steelblue', label=" Training time", linewidth=2.0)
ax.plot(X, TRAIN_TIME, color='steelblue', marker='o', markersize=4)

ax.legend(loc='best loc', frameon=True)
plt.grid(True, linestyle='--')
plt.title("Computational time , Randomized projections")
plt.xlabel(" Number of componenets")
plt.ylabel(" Time (seconds)")
plt.savefig('plots/NN_RP_time.png', bbox_inches='tight')

transformer = FastICA(n_components=40, random_state=111, tol=0.001)
x_train_transform = transformer.fit_transform(x_train)
x_test_transform = transformer.transform(x_test)

rng = np.random.RandomState(111)
transformer = SparseRandomProjection(random_state=rng, n_components=350)
x_train_transform = transformer.fit_transform(x_train)
x_test_transform = transformer.transform(x_test)
n_nodes = 11
n_comps = 350
model_name = "CNN"
model = Sequential()
model.add(Reshape(target_shape=(1, 1, n_comps), input_shape=(n_comps,)))
model.add(Flatten())
model.add(Dense(output_dim=64, activation='relu'))
model.add(Dense(output_dim=10, activation='softmax'))
adam = keras.optimizers.Adam(lr=0.01, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])

model.fit(x_train_transform, y_train, epochs=150, batch_size=128, verbose=False)

y_pred_train_cv = model.predict_classes(x_train_transform)
y_pred_test_cv = model.predict_classes(x_test_transform)

accuracy_train_cv = score(from_onehot(y_train), y_pred_train_cv, "accuracy")
accuracy_test_cv = score(from_onehot(y_test), y_pred_test_cv, "accuracy")

time1 = time.time()
TIME_ARR.append(time1 - time0)
TRAIN_ARR.append(accuracy_train_cv)
TEST_ARR.append(accuracy_test_cv)
print("Randomized projections, n_comps : ",   n_comps, "accuracy_train_cv: ", accuracy_train_cv, "accuracy_test_cv: ", accuracy_test_cv, "Time ",
      time1 - time0)


"""

TruncatedSVD

"""


print("\n\n\n\n Truncated SVD")
from sklearn import random_projection
from sklearn.decomposition import TruncatedSVD
from sklearn.random_projection import sparse_random_matrix
N_COMPS = np.arange(1,40,1)
TRAIN_ARR = []
TEST_ARR  = []
TIME_ARR  = []
"""
for n_comps in N_COMPS:
    transformer = TruncatedSVD(n_components=n_comps, n_iter=10, random_state=111)
    x_train_transform = transformer.fit_transform(x_train_cv)
    x_test_transform  = transformer.transform(x_test_cv)

    time0 = time.time()

    n_nodes = 11
    model_name = "CNN"
    model = Sequential()
    model.add(Reshape(target_shape=(1, 1, n_comps), input_shape=(n_comps,)))
    model.add(Flatten())
    model.add(Dense(output_dim=64, activation='relu'))
    model.add(Dense(output_dim=10, activation='softmax'))
    adam = keras.optimizers.Adam(lr=0.01, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])

    model.fit(x_train_transform, y_train_cv, epochs=150, batch_size=128, verbose = False)

    y_pred_train_cv = model.predict_classes(x_train_transform)
    y_pred_test_cv = model.predict_classes(x_test_transform)

    accuracy_train_cv = score(from_onehot(y_train_cv), y_pred_train_cv, "accuracy")
    accuracy_test_cv = score(from_onehot(y_test_cv), y_pred_test_cv, "accuracy")


    time1 = time.time()
    TIME_ARR.append(time1-time0)
    TRAIN_ARR.append(accuracy_train_cv)
    TEST_ARR.append(accuracy_test_cv)
    print(n_comps, "accuracy_train_cv: ", accuracy_train_cv, "accuracy_test_cv: ", accuracy_test_cv, "Time ",
          time1 - time0)


np.save("data/NN_spPCA_train.npy",TRAIN_ARR)
np.save("data/NN_spPCA_test.npy",TEST_ARR)
np.save("data/NN_spPCA_time.npy", TIME_ARR)
"""
TRAIN_ACCURACY = np.load('data/NN_spPCA_train.npy', allow_pickle=True)
TEST_ACCURACY  = np.load('data/NN_spPCA_test.npy', allow_pickle=True)
TRAIN_TIME     = np.load('data/NN_spPCA_time.npy', allow_pickle=True)


fig, ax = plt.subplots()
X = N_COMPS
#ax.plot(X, TRAIN_ACCURACY, color='steelblue', label=" Training accuracy.", linewidth=2.0)
#ax.plot(X, TRAIN_ACCURACY, color='steelblue', marker='o', markersize=4)

ax.plot(X, TEST_ACCURACY, color='red', label="Validation accuracy.", linewidth=2.0)
ax.plot(X, TEST_ACCURACY, color='red', marker='o', markersize=4)

ax.legend(loc='best loc', frameon=True)
plt.grid(True, linestyle='--')
plt.title("Validation curve, Truncated SVD")
plt.xlabel(" Number of components")
plt.ylabel(" Accuracy ")
plt.savefig('plots/NN_SVD_acc.png', bbox_inches='tight')



fig, ax = plt.subplots()
X = N_COMPS
ax.plot(X, TRAIN_TIME, color='steelblue', label=" Training time", linewidth=2.0)
ax.plot(X, TRAIN_TIME, color='steelblue', marker='o', markersize=4)

ax.legend(loc='best loc', frameon=True)
plt.grid(True, linestyle='--')
plt.title("Computational time , Truncated SVD")
plt.xlabel(" Number of componenets")
plt.ylabel(" Time (seconds)")
plt.savefig('plots/NN_SVD_time.png', bbox_inches='tight')

transformer = FastICA(n_components=50, random_state=111, tol=0.001)
x_train_transform = transformer.fit_transform(x_train)
x_test_transform = transformer.transform(x_test)

n_nodes = 11
n_comps = 50
model_name = "CNN"
model = Sequential()
model.add(Reshape(target_shape=(1, 1, n_comps), input_shape=(n_comps,)))
model.add(Flatten())
model.add(Dense(output_dim=64, activation='relu'))
model.add(Dense(output_dim=10, activation='softmax'))
adam = keras.optimizers.Adam(lr=0.01, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])

model.fit(x_train_transform, y_train, epochs=150, batch_size=128, verbose=False)

y_pred_train_cv = model.predict_classes(x_train_transform)
y_pred_test_cv = model.predict_classes(x_test_transform)

accuracy_train_cv = score(from_onehot(y_train), y_pred_train_cv, "accuracy")
accuracy_test_cv = score(from_onehot(y_test), y_pred_test_cv, "accuracy")

time1 = time.time()
TIME_ARR.append(time1 - time0)
TRAIN_ARR.append(accuracy_train_cv)
TEST_ARR.append(accuracy_test_cv)
print("Kernel PCA, n_comps : ",   n_comps, "accuracy_train_cv: ", accuracy_train_cv, "accuracy_test_cv: ", accuracy_test_cv, "Time ",
      time1 - time0)

