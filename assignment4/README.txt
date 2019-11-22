#  Project4. Markov Decision Processes.
# Gatech, CS7641.


The link to the code: https://github.com/maksimkazanskii/CS7641/tree/master/assignment4

## Getting Started and Installation

1. Download the source code /Assignment4/ to the working folder. 


## Prerequisites
### Working kernel:
Python 3.7

### Necessary libraries:
sklearn, 
numpy,
pandas,
time,
random, 
seaborn,
matplotlib,
os,
pprint,
gym        ( http://gym.openai.com/docs/),
mdptoolbox ( https://pymdptoolbox.readthedocs.io/en/latest/api/mdptoolbox.html )

I anticipate that the listed libraries are installed, if not for please refer to the proper resources. For some libraries the documentation pages are indicated in the braces.
If there are any questions please do not hesitate to contact me. 

## Files


### MNIST folder:
* /data/                      - empty folder with precomputed and saved numpy arrays for the plotting
* /plots/                     - empty folder with output plots
* Forest_policy_iteration.py  - policy iteration algorithm for Forest MDP
* Forest_Qlearning.py         - Q learning algorithm for Forest MDP
* Forest_viteration.py        - value iteration algorithm for Forest MDP
* Lake_policy_iteration.py    - policy iteration algorithm for Frozen Lake MDP
* Lake_Qlearning.py           - Q learning algorithm for Frozen Lake MDP
* Lake_viteration.py          - value iteration algorithm for Frozen Lake MDP



## Running the tests:
In order to run the tests you need to execute each program manually. For example, for executing the experiments with Forest MDP and value iteration please run the following script in the terminal:

**python Forest_viteration.py**


## Code Citations

* Calculating moving average           :  https://stackoverflow.com/questions/14313510/how-to-calculate-moving-average-using-numpy
* Basic Q learning implementation      :  https://github.com/simoninithomas/Deep_reinforcement_learning_Course/blob/master/Q%20learning/FrozenLake/Q%20Learning%20with%20FrozenLake.ipynb 
* Basic policy iteration implementaion :  https://github.com/dennybritz/reinforcement-learning/blob/master/DP/Policy%20Iteration%20Solution.ipynb
* Basic value iteration implementation :  https://github.com/dennybritz/reinforcement-learning/blob/master/DP/Value%20Iteration%20Solution.ipynb

## Authors

* **Maksim Kazanskii** 

