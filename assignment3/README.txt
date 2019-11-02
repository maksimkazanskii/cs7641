#  Project1. Supervised learning.
# Gatech, CS7641.


The link to the code: https://github.com/maksimkazanskii/CS7641/tree/master/assignment3

## Getting Started and Installation

1. Download the source code /Assignment3/ to the working folder. 
2. Upload the MNIST file (mnist_train.csv) from https://www.dropbox.com/s/h8ysynlkk0t9zb2/mnist_train.csv?dl=0 to /Assignment3/data/. This complication is due to the github file size limit.

## Prerequisites
### Working kernel:
Python 3.7

### Necessary libraries:
sklearn, 
numpy,
pandas,
keras,
tensorflow,
time,
random, 
seaborn,
matplotlib,
os,
pprint

I anticipate that the listed libraries are installed, if not for please refer to the proper resources. If there are any questions please do not hesitate to contact me. 

## Files


### MNIST folder:
* /data/                  - folder containing datasets for diabetes and MNIST dataset and precomputed numpy arrays.
* /plots/                 - empty folder with output plots.
* data_prep.py            - data preprocessing helper file.
* diabetes_clustering.py  - clustering experiments for diabetes dataset.
* diabetes_ICA.py         - ICA analysis  for diabetes dataset.
* diabetes_PCA.py         - PCA analysis  for diabetes dataset.
* diabetes_RP.py          - RP analysis  for diabetes dataset.
* diabetes_SVD.py         - SVD analysis  for diabetes dataset.
* MNIST_ICA.py            - ICA analysis  for MNIST dataset.
* MNIST_Kmeans.py         - Kmeans clustering for MNIST dataset.
* MNIST_me.py             - Expectation MAximization clustering for MNIST dataset
* MNIST_RP.py             - RP analysis  for MNIST dataset.
* MNIST_SVD.py            - SVD analysis  for MNIST dataset.
* NN_clustering.py        - Clustering experiments with the Neural Network.
* NN_dim.py		  - Dimensionality reduction  experiments for Nerual network.



## Running the tests:
In order to run the tests you need to execute each program manually. For example, for executing the experiments with Neural Networks and dimensionality reduction on the MNIST dataset you need to run the following command on the terminal:

**python NN_dim.py**


Please note, that the execution time for each of the algorithms is sifficient. Overall, it takes around 1 hour for Neural Network weight optimization to run (4 core processor with NVIDIA) mostly due to genetic algorithm.

Important: Some of the functions are specifically commented in order to reduce the computational time. The corresponding numpy arrays are located in /data/ folder. If you want to fully run the program please uncomment the specific parts inside of the programs where it is necessary.

## Code Citations

* Neural network architecture(MNIST): https://www.kaggle.com/endlesslethe/siwei-digit-recognizer-top20
* Plots : https://matplotlib.org/3.1.1/gallery/images_contours_and_fields/image_annotated_heatmap.html
* Plots :  https://scikit-learn.org/stable/auto_examples/model_selection/plot_learning_curve.html
* Confusion matrix plots:  https://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html#sphx-glr-auto-examples-model-selection-plot-confusion-matrix-py
* MNIST data:          https://www.kaggle.com/oddrationale/mnist-in-csv
* calcualting f-score in keras : https://stackoverflow.com/questions/43547402/how-to-calculate-f1-macro-in-keras
* Inverse transform - https://stackoverflow.com/questions/32750915/pca-inverse-transform-manually
* Inverse transform - https://scprep.readthedocs.io/en/stable/_modules/scprep/reduce.html

## Authors

* **Maksim Kazanskii** 

