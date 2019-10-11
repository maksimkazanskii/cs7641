import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import myplots
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import ShuffleSplit
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix
import time
import keras
import mlrose
from sklearn.preprocessing import MinMaxScaler , OneHotEncoder


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
        return accuracy_score(y_true, y_pred)

def from_onehot(g):

    vector = np.where(g == 1)[1]
    return vector

def all_score(y_true, y_pred):

    precision = precision_score(y_true, y_pred, average='macro')
    recall = recall_score(y_true, y_pred, average = 'macro')
    accuracy = accuracy_score(y_true, y_pred)
    return (accuracy, precision, recall)

np.random.seed(111)


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

# scaler = MinMaxScaler()
# x_train_cv = scaler.fit_transform(x_train_cv)
# x_test_cv  = scaler.transform(x_test_cv)

y_all_pred = np.zeros((3, n_samples_test)).astype(np.int64)
print(y_all_pred.dtype)

# CNN nnodes hyperparameter tuning - Learning rate


acc_train_cv_arr = []
acc_test_cv_arr  = []


LEARNING_RATE = [0.01]
# Uncomment to run the code

history = []
TIME = []
TRAIN_ACCURACY = []
TEST_ACCURACY  = []
"""
PROB = [0.01,0.0033,0.001]

for prob in PROB:

    time0 = time.time()

    model = mlrose.NeuralNetwork(hidden_nodes=[64], activation='relu', algorithm='genetic_alg',
                                 max_iters=1000, bias=False, is_classifier=True, learning_rate=1.0,
                                 early_stopping=False, clip_max=10.0, restarts=0,
                                 pop_size=200, mutation_prob=prob, max_attempts=1,
                                 random_state=None, curve = True)

    history.append(model.fit(x_train_cv, y_train_cv))
    time1 = time.time()
    y_train_pred = model.predict(x_train_cv)
    train_accuracy = accuracy_score(y_train_cv, y_train_pred)
    TRAIN_ACCURACY.append(train_accuracy)
    y_test_pred = model.predict(x_test_cv)
    test_accuracy = accuracy_score(y_test_pred, y_test_cv)
    TEST_ACCURACY.append(test_accuracy)
    print("prob, Train accuracy, Test accuracy ,time:", prob, train_accuracy, test_accuracy, time1 - time0)
    TIME.append(time1-time0)

TRAIN_ACCURACY = np.array(TRAIN_ACCURACY)
TEST_ACCURACY  =np.array(TEST_ACCURACY)
FIT_ARR = []
for i in range(len(history)):
    FIT_ARR.append(history[i].fitness_curve)
FIT_ARR = np.array(FIT_ARR)
TIME = np.array(TIME)
"""
#POP_SIZE = [200]
#POP_SIZE = [20,50,100]
PROBS  = [0.01,0.0033,0.001]
for prob  in PROBS:
    time0 = time.time()
    model = mlrose.NeuralNetwork(hidden_nodes=[64], activation='relu', algorithm='genetic_alg',
                                 max_iters=5000, bias=True, is_classifier=True, learning_rate=1.0,
                                 early_stopping=False, clip_max=10.0, restarts=0,
                                 pop_size=200, mutation_prob=0.001, max_attempts=100,
                                 random_state=None, curve = True)

    history.append(model.fit(x_train_cv, y_train_cv))
    time1 = time.time()
    y_train_pred = model.predict(x_train_cv)
    train_accuracy = accuracy_score(y_train_cv, y_train_pred)
    TRAIN_ACCURACY.append(train_accuracy)
    y_test_pred = model.predict(x_test_cv)
    test_accuracy = accuracy_score(y_test_pred, y_test_cv)
    TEST_ACCURACY.append(test_accuracy)
    print("prob, Train accuracy, Test accuracy ,time:", prob, train_accuracy, test_accuracy, time1 - time0)
    TIME.append(time1-time0)

TRAIN_ACCURACY = np.array(TRAIN_ACCURACY)
TEST_ACCURACY  =np.array(TEST_ACCURACY)
FIT_ARR = []
for i in range(len(history)):
    FIT_ARR.append(history[i].fitness_curve)
FIT_ARR = np.array(FIT_ARR)
TIME = np.array(TIME)

np.save("data/NN_GA_TRAIN1.npy", TRAIN_ACCURACY)
np.save("data/NN_GA_TEST1.npy",  TEST_ACCURACY )
np.save("data/NN_GA_FIT1.npy",   FIT_ARR)
np.save("data/NN_GA_TIME1.npy",  TIME)

TRAIN_ACCURACY = np.load('data/NN_GA_TRAIN1.npy')
TEST_ACCURACY  = np.load('data/NN_GA_TEST1.npy')
FIT_ARR        = np.load('data/NN_GA_FIT1.npy')
TIME           = np.load('data/NN_GA_TIME1.npy')


X0 = np.arange(0,len(FIT_ARR[0]),1)
X1 = np.arange(0,len(FIT_ARR[1]),1)
X2 = np.arange(0,len(FIT_ARR[2]),1)


fig, ax = plt.subplots()
ax.plot(X0, -FIT_ARR[1], color='steelblue', label="Loss curve (prob = 0.01)", linewidth=2.0, linestyle='-')
ax.plot(X0, -FIT_ARR[0], color='red', label="Loss curve (prob = 0.0033) ", linewidth=2.0, linestyle='-')
ax.plot(X0, -FIT_ARR[2], color='black', label="Loss curve ( prob = 0.001)", linewidth=2.0, linestyle='-')

ax.legend(loc='best', frameon=True)
plt.grid(True, linestyle='--')
plt.title('Genetic algorithm, Loss vs Number of iteration.\n ')
plt.ylabel('Loss')
plt.xlabel('Iterations')
plt.savefig('plots/NN_Genetic2.png')

"""
fig, ax = plt.subplots()
X = np.log(np.array(LEARNING_RATE))
ax.plot(X, TRAIN_ACCURACY, color='steelblue', label=" Training accuracy.", linewidth=2.0)
ax.plot(X, TRAIN_ACCURACY, color='steelblue', marker='o', markersize=4)

ax.plot(X, TEST_ACCURACY, color='red', label="Validation accuracy.", linewidth=2.0)
ax.plot(X, TEST_ACCURACY, color='red', marker='o', markersize=4)

ax.legend(loc='best loc', frameon=True)
plt.grid(True, linestyle='--')
plt.title("Performance of the NN (Gradient Descent) for different learning rates")
plt.xlabel(" Log_10(Lr)")
plt.ylabel(" Accuracy ")
plt.savefig('plots/NN_GA2.png')

"""









