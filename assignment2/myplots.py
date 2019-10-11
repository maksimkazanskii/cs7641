import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels
import numpy as np
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.datasets import load_digits
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit

def hp_plot(X, Y1,Y2, legend1, legend2, title, xlabel, ylabel, figurename):
    fig, ax = plt.subplots()
    ax.plot(X, Y1, color = 'steelblue',  label=legend1, linewidth = 2.0)
    ax.plot(X, Y1, color = 'steelblue',  marker='o', markersize = 4)
    ax.plot(X, Y2,  color ='red',     label=legend2, linewidth = 2.0)
    ax.plot(X, Y2,  color ='red',     marker='o', markersize = 4)
    ax.legend(loc='best loc', frameon=True)
    plt.grid(True,linestyle='--')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.savefig(figurename)




def hp_plot_mean(x_arr, train_scores,test_scores, label1, label2, title, labelX, labelY, filename):
    fig, ax = plt.subplots()
    tr_mean = train_scores.mean(axis = 1)
    tr_std  = train_scores.std(axis =1)
    test_mean = test_scores.mean(axis = 1)
    test_std  = test_scores.std(axis =1)
    ax.plot(x_arr, tr_mean, color='steelblue', label=label1, linewidth=2.0)
    ax.plot(x_arr, tr_mean, color='steelblue', marker='o', markersize=4)
    #ax.fill_between(x_arr, tr_mean - tr_std,
    #                tr_mean + tr_std, alpha=0.2,
    #                color="orange")
    ax.plot(x_arr, test_mean, color='red', label=label2, linewidth=2.0)
    ax.plot(x_arr, test_mean, color='red', marker='o', markersize=4)

    #ax.fill_between(x_arr, test_mean - test_std,
    #                test_mean+test_std, alpha=0.2,
    #                color="red")
    ax.legend(loc='best', frameon=True)
    plt.title(title)
    plt.grid(True, linestyle='--')
    plt.xlabel(labelX)
    plt.ylabel(labelY)
    plt.savefig(filename)

"""
myplots.hp_plot_mean_mlt(N_NODES, [acc_train_cv_arr_lr1,acc_train_cv_arr_lr2, acc_train_cv_arr_lr2],
                         [acc_test_cv_arr_lr1,acc_test_cv_arr_lr2, acc_test_cv_arr_lr2],
                         ["Training accuracy ( lr= 0.001)", "Training accuracy ( lr= 0.0005)","Training accuracy ( lr= 0.0001)"],
                         ["Testing accuracy ( lr= 0.001)", "Testing accuracy ( lr= 0.0005)","Testing accuracy ( lr= 0.0001)"],
                         "ANN,Accuracy vs Number of Nodes (different Leaning rates)", "Number of Nodes", "Accruacy","plots/NN5.png")
"""


def hp_plot_mean_mlt(x_arr, train_scores,test_scores, label_arr_train, label_arr_test, title, labelX, labelY, filename):
    fig, ax = plt.subplots()
    tr_mean1 = train_scores[0].mean()
    test_mean1 = test_scores[0].mean()


    tr_mean2 = train_scores[1].mean()
    test_mean2 = test_scores[1].mean()


    tr_mean3 = train_scores[2].mean()
    test_mean3 = test_scores[2].mean()

    #linestyles = ['-', '--', '-.', ':']
    ax.plot(x_arr, tr_mean1, color='steelblue', label=label_arr_train[0], linewidth=2.0,linestyle='-')
    ax.plot(x_arr, tr_mean1, color='steelblue', marker='o', markersize=4)
    ax.plot(x_arr, test_mean1, color='red', label=label_arr_test[0], linewidth=2.0,linestyle ='-')
    ax.plot(x_arr, test_mean1, color='red', marker='o', markersize=4)

    ax.plot(x_arr, tr_mean1, color='steelblue', label=label_arr_train[1], linewidth=2.0, linestyle='--')
    ax.plot(x_arr, tr_mean1, color='steelblue', marker='o', markersize=4)
    ax.plot(x_arr, test_mean1, color='red', label=label_arr_test[1], linewidth=2.0, linestyle='--')
    ax.plot(x_arr, test_mean1, color='red', marker='o', markersize=4)

    ax.plot(x_arr, tr_mean1, color='steelblue', label=label_arr_train[2], linewidth=2.0, linestyle=':')
    ax.plot(x_arr, tr_mean1, color='steelblue', marker='o', markersize=4)
    ax.plot(x_arr, test_mean1, color='red', label=label_arr_test[2], linewidth=2.0, linestyle=':')
    ax.plot(x_arr, test_mean1, color='red', marker='o', markersize=4)

    ax.legend(loc='best', frameon=True)
    plt.title(title)
    plt.grid(True, linestyle='--')
    plt.xlabel(labelX)
    plt.ylabel(labelY)
    plt.savefig(filename)


def hp_plot_var(X, Y1,Y2, dY1,dY2, legend1, legend2, title, xlabel, ylabel, figurename):
    fig, ax = plt.subplots()
    ax.plot(X, Y1, color = 'steelblue',  label=legend1, linewidth = 2.0)
    ax.plot(X, Y1, color = 'steelblue',  marker='o', markersize = 4)
    ax.fill_between(X, Y1 - dY1,
                    Y1 + dY1, alpha=0.2,
                    color="steelblue")
    ax.plot(X, Y2,  color ='red',     label=legend2, linewidth = 2.0)
    ax.plot(X, Y2,  color ='red',     marker='o', markersize = 4)
    ax.fill_between(X, Y2 - dY2,
                    Y2 + dY2, alpha=0.2,
                    color="red")
    ax.legend(loc='best loc', frameon=True)
    plt.grid(True,linestyle='--')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.savefig(figurename)


def plot_history_NN(history, filename):
    fig, ax = plt.subplots()
    ax.plot(history.history['acc'],color = 'steelblue',  label="Training accuracy", linewidth = 2.0)
    ax.plot(history.history['val_acc'], color = 'red',  label="Validation accuracy", linewidth = 2.0)
    plt.legend(['Training accuracy', 'Validation accuracy'], loc='lower right', frameon =True)
    plt.grid(True, linestyle='--')
    plt.title('ANN, Accuracy vs Number of Epochs')
    plt.ylabel('Accuracy')
    plt.xlabel('Epochs')
    plt.savefig(filename)



def plot_history_NN_arr(history_arr,label_arr_train, label_arr_test, filename):
    fig, ax = plt.subplots()

    ax.plot(history_arr[0].history['acc'],color = 'steelblue',  label=label_arr_train[0], linewidth = 1.0, linestyle='-')
    ax.plot(history_arr[0].history['val_acc'], color = 'steelblue',  label=label_arr_test[0], linewidth = 1.0, linestyle=':')

    ax.plot(history_arr[1].history['acc'], color='red', label=label_arr_train[1], linewidth=1.0, linestyle='-')
    ax.plot(history_arr[1].history['val_acc'], color='red', label=label_arr_test[1], linewidth=1.0, linestyle=':')

    ax.plot(history_arr[2].history['acc'], color='black', label=label_arr_train[2], linewidth=1.0, linestyle='-')
    ax.plot(history_arr[2].history['val_acc'], color='black', label=label_arr_test[2], linewidth=1.0, linestyle=':')
    ax.legend(loc='best', frameon=True)


    plt.grid(True, linestyle='--')
    plt.title('ANN, Accuracy vs Number of Epochs')
    plt.ylabel('Accuracy')
    plt.xlabel('Epochs')
    plt.savefig(filename)


def plot_history_boost_arr(X,Y_train,Y_test,label_arr_train, label_arr_test, filename):
    fig, ax = plt.subplots()

    ax.plot(X,Y_train[0], color = 'steelblue',  label=label_arr_train[0], linewidth = 1.0, linestyle='-')
    ax.plot(X,Y_test[0], color = 'steelblue',  label=label_arr_test[0], linewidth = 1.0, linestyle='--')

    ax.plot(X,Y_train[1], color='red', label=label_arr_train[1], linewidth=1.0, linestyle='-')
    ax.plot(X,Y_test[1], color='red', label=label_arr_test[1], linewidth=1.0, linestyle='--')

    ax.plot(X,Y_train[2], color='black', label=label_arr_train[2], linewidth=1.0, linestyle='-')
    ax.plot(X,Y_test[2], color='black', label=label_arr_test[2], linewidth=1.0, linestyle='--')
    ax.legend(loc='best', frameon=True)

    plt.grid(True, linestyle='--')
    plt.title('AdaBoost, Accuracy vs Number of iterations')
    plt.ylabel('Accuracy')
    plt.xlabel('Number of iterations')
    plt.savefig(filename)


def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=None, train_sizes=np.linspace(.05, 1.0, 5), filename = "default.png"):
    """
    Generate a simple plot of the test and training learning curve.

    Parameters
    ----------
    estimator : object type that implements the "fit" and "predict" methods
        An object of that type which is cloned for each validation.

    title : string
        Title for the chart.

    X : array-like, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.

    y : array-like, shape (n_samples) or (n_samples, n_features), optional
        Target relative to X for classification or regression;
        None for unsupervised learning.

    ylim : tuple, shape (ymin, ymax), optional
        Defines minimum and maximum yvalues plotted.

    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:
          - None, to use the default 3-fold cross-validation,
          - integer, to specify the number of folds.
          - :term:`CV splitter`,
          - An iterable yielding (train, test) splits as arrays of indices.

        For integer/None inputs, if ``y`` is binary or multiclass,
        :class:`StratifiedKFold` used. If the estimator is not a classifier
        or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validators that can be used here.

    n_jobs : int or None, optional (default=None)
        Number of jobs to run in parallel.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    train_sizes : array-like, shape (n_ticks,), dtype float or int
        Relative or absolute numbers of training examples that will be used to
        generate the learning curve. If the dtype is float, it is regarded as a
        fraction of the maximum size of the training set (that is determined
        by the selected validation method), i.e. it has to be within (0, 1].
        Otherwise it is interpreted as absolute sizes of the training sets.
        Note that for classification the number of samples usually have to
        be big enough to contain at least one sample from each class.
        (default: np.linspace(0.1, 1.0, 5))
    """
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Accuracy")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    plt.grid(True, linestyle='--')
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.2,
                     color="steelblue")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.2, color="red")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="steelblue", markersize = 4,
             label="Training accuracy")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="red", markersize = 4,
             label="CV accuracy")

    plt.legend(loc="best")
    plt.savefig(filename)



def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues,
                          filename='default.png'):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    plt.savefig(filename)


# Taken from https://matplotlib.org/3.1.1/gallery/images_contours_and_fields/image_annotated_heatmap.html

def heatmap(data, row_labels, col_labels, ax=None,
            cbar_kw={}, cbarlabel="", **kwargs):
    """
    Create a heatmap from a numpy array and two lists of labels.

    Parameters
    ----------
    data
        A 2D numpy array of shape (N, M).
    row_labels
        A list or array of length N with the labels for the rows.
    col_labels
        A list or array of length M with the labels for the columns.
    ax
        A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
        not provided, use current axes or create a new one.  Optional.
    cbar_kw
        A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
    cbarlabel
        The label for the colorbar.  Optional.
    **kwargs
        All other arguments are forwarded to `imshow`.
    """

    if not ax:
        ax = plt.gca()

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    # We want to show all ticks...
    ax.set_xticks(np.arange(data.shape[1]))
    ax.set_yticks(np.arange(data.shape[0]))
    # ... and label them with the respective list entries.
    ax.set_xticklabels(col_labels)
    ax.set_yticklabels(row_labels)

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=True, bottom=False,
                   labeltop=True, labelbottom=False)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=-30, ha="right",
             rotation_mode="anchor")

    # Turn spines off and create white grid.
    for edge, spine in ax.spines.items():
        spine.set_visible(False)

    ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im, cbar


def annotate_heatmap(im, data=None, valfmt="{x:.2f}",
                     textcolors=["black", "white"],
                     threshold=None, **textkw):
    """
    A function to annotate a heatmap.

    Parameters
    ----------
    im
        The AxesImage to be labeled.
    data
        Data used to annotate.  If None, the image's data is used.  Optional.
    valfmt
        The format of the annotations inside the heatmap.  This should either
        use the string format method, e.g. "$ {x:.2f}", or be a
        `matplotlib.ticker.Formatter`.  Optional.
    textcolors
        A list or array of two color specifications.  The first is used for
        values below a threshold, the second for those above.  Optional.
    threshold
        Value in data units according to which the colors from textcolors are
        applied.  If None (the default) uses the middle of the colormap as
        separation.  Optional.
    **kwargs
        All other arguments are forwarded to each call to `text` used to create
        the text labels.
    """

    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max())/2.

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(horizontalalignment="center",
              verticalalignment="center")
    kw.update(textkw)

    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
            text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
            texts.append(text)

    return texts