
# from https://www.kaggle.com/endlesslethe/siwei-digit-recognizer-top20

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
import os
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
import random

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

def get_data_MNIST():
    np.random.seed(1111)
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

    return x,y


def get_data_mall():
    file = open("data/mnist_train.csv")
    data = pd.read_csv(file)

    x = np.array(data.iloc[:])
    return x
def main():
    _,Y = get_data_MNIST()
    print("benchmark_accuracy: ", np.count_nonzero(Y == 1)/ Y.shape[0])
    numbers = np.arange(11)
    n, bins, patches = plt.hist(Y,facecolor='steelblue', alpha=0.75,rwidth =0.9,bins=np.arange(11)-0.5)
    plt.xticks(numbers, numbers)
    plt.xlabel('Number')
    plt.ylabel('Distribution')
    plt.title('Multiclass distribution')
    plt.savefig("plots/dits.png")



if __name__ == "__main__":
    main()