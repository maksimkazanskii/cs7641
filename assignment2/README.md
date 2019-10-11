#  Project1. Supervised learning.
# Gatech, CS7641.


The link to the code: https://github.com/maksimkazanskii/cs7641/tree/master/assignment2

## Getting Started and Installation

1. Download the source code /Assignment2/ to the working folder. 
2. Upload the MNIST file (mnist_train.csv) from https://www.dropbox.com/s/h8ysynlkk0t9zb2/mnist_train.csv?dl=0 to /Assignment2/data/. This complication is due to the github file size limit.
3. Install MLRose library : pip install mlrose
4. Move the python files from /Assignment2/mlrose_custom/ to the library folder of installed MLRose library (usually - ../lib/python3.7/site-packages/mlrose/). This is due to the changes in the MLRose code. 

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
* fourpeaks.py  - optimization algorithms for forupeaks problem
* knapsack.py   - optimization algorithms for knapsack problem
* sumeven.py    - optimization algorithms for sumeven problem
* myplots.py    - customized plots
* NN_GA.py      - Neural network weights optimization for genetic algorithm
* NN_regular.py - Neural network weights optimization for backpropagation algorithm
* NN_SA.py      - Neural network weights optimization for simulated annealing algorithm
* NN_RHC.py     - Neural network weights optimization for random hill climbing algorithm

### other folders : 
* plots  - empty folder with output plots.

## Running the tests:
In order to run the tests you need to execute each program manually. For example, for executing the experiments with Neural Networks on the MNIST dataset you need to run the following command on the terminal:

**python NN_GA.py**


Please note, that the execution time for each of the algorithms is sifficient. Overall, it takes around 24 hours for Neural Network weight optimization to run (4 core processor with NVIDIA) mostly due to genetic algorithm.

The execution time forspecific problem (fourpeaks, knapsack and sumeven) should be less than hour.

## Code Citations

* Neural network architecture(MNIST): https://www.kaggle.com/endlesslethe/siwei-digit-recognizer-top20
* Plots : https://matplotlib.org/3.1.1/gallery/images_contours_and_fields/image_annotated_heatmap.html
* Plots :  https://scikit-learn.org/stable/auto_examples/model_selection/plot_learning_curve.html
* Confusion matrix plots:  https://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html#sphx-glr-auto-examples-model-selection-plot-confusion-matrix-py
* MNIST data:          https://www.kaggle.com/oddrationale/mnist-in-csv
* calcualting f-score in keras : https://stackoverflow.com/questions/43547402/how-to-calculate-f1-macro-in-keras
* Heatmap plot : https://matplotlib.org/3.1.1/gallery/images_contours_and_fields/image_annotated_heatmap.html
* MLRose library : https://pypi.org/project/mlrose/
## Authors

* **Maksim Kazanskii** 

