import mlrose
import numpy as np
import myplots
import matplotlib.pyplot as plt
import time
import math
# https://mlrose.readthedocs.io/en/stable/source/tutorial1.html

# Define alternative N-Queens fitness function for maximization problem


def head(_b, _x):
    """Determine the number of leading b's in vector x.
    Parameters
    ----------
    b: int
        Integer for counting at head of vector.
    x: array
        Vector of integers.
    Returns
    -------
    head: int
        Number of leading b's in x.
    """

    # Initialize counter
    _head = 0

    # Iterate through values in vector
    for i in _x:
        if i == _b:
            _head += 1
        else:
            break

    return _head


def tail(_b, _x):
    """Determine the number of trailing b's in vector x.
    Parameters
    ----------
    b: int
        Integer for counting at tail of vector.
    x: array
        Vector of integers.
    Returns
    -------
    tail: int
        Number of trailing b's in x.
    """

    # Initialize counter
    _tail = 0

    # Iterate backwards through values in vector
    for i in range(len(_x)):
        if _x[len(_x) - i - 1] == _b:
            _tail += 1
        else:
            break

    return _tail


def my_fitness(state):
    """Evaluate the fitness of a state vector.
    Parameters
    ----------
    state: array
        State array for evaluation.
    Returns
    -------
    fitness: float.
        Value of fitness function.
    """
    _n = len(state)
    _t = np.ceil(0.1 * _n)

    # Calculate head and tail values
    tail_0 = tail(0, state)
    head_1 = head(1, state)

    # Calculate R(X, T)
    if (tail_0 > _t and head_1 > _t):
        _r = _n
    else:
        _r = 0

    # Evaluate function
    fitness = max(tail_0, head_1) + _r
    FIT_ARR.append(fitness)
    TIME_ARR.append(time.time())
    return fitness

def extend(arr, max):
    last = arr[-1]
    a = np.empty(max)
    a.fill(last)
    a[0:len(arr)]=arr
    return a

def my_best_fitness(arr):
    print("Erger")
    return np.maximum.accumulate(arr)

def my_reach_max(arr,value):
    args_max= np.argwhere(arr==value)
    if len(args_max)==0:
        return None
    else:
        return args_max[0][0]/1000.0

def extend(arr,length):
    new_arr = np.empty(length)
    new_arr.fill(arr[-1])
    new_arr[0:len(arr)] = np.copy(arr)
    return new_arr


class my_Knapsack:
    """Fitness function for Knapsack optimization problem. Given a set of n
    items, where item i has known weight :math:`w_{i}` and known value
    :math:`v_{i}`; and maximum knapsack capacity, :math:`W`, the Knapsack
    fitness function evaluates the fitness of a state vector
    :math:`x = [x_{0}, x_{1}, \\ldots, x_{n-1}]` as:
    .. math::
        Fitness(x) = \\sum_{i = 0}^{n-1}v_{i}x_{i}, \\text{ if}
        \\sum_{i = 0}^{n-1}w_{i}x_{i} \\leq W, \\text{ and 0, otherwise,}
    where :math:`x_{i}` denotes the number of copies of item i included in the
    knapsack.
    Parameters
    ----------
    weights: list
        List of weights for each of the n items.
    values: list
        List of values for each of the n items.
    max_weight_pct: float, default: 0.35
        Parameter used to set maximum capacity of knapsack (W) as a percentage
        of the total of the weights list
        (:math:`W =` max_weight_pct :math:`\\times` total_weight).
    Example
    -------
    .. highlight:: python
    .. code-block:: python

    Note
    ----
    The Knapsack fitness function is suitable for use in discrete-state
    optimization problems *only*.
    """

    def __init__(self, weights, values, max_weight_pct=0.35):

        self.weights = weights
        self.values = values
        self._w = np.ceil(np.sum(self.weights)*max_weight_pct)
        self.prob_type = 'discrete'

        if len(self.weights) != len(self.values):
            raise Exception("""The weights array and values array must be"""
                            + """ the same size.""")

        if min(self.weights) <= 0:
            raise Exception("""All weights must be greater than 0.""")

        if min(self.values) <= 0:
            raise Exception("""All values must be greater than 0.""")

        if max_weight_pct <= 0:
            raise Exception("""max_weight_pct must be greater than 0.""")

    def evaluate(self, state):
        """Evaluate the fitness of a state vector.
        Parameters
        ----------
        state: array
            State array for evaluation. Must be the same length as the weights
            and values arrays.
        Returns
        -------
        fitness: float
            Value of fitness function.
        """

        if len(state) != len(self.weights):
            raise Exception("""The state array must be the same size as the"""
                            + """ weight and values arrays.""")

        # Calculate total weight and value of knapsack
        total_weight = np.sum(state*self.weights)
        total_value = np.sum(state*self.values)

        # Allow for weight constraint
        if total_weight <= self._w:
            fitness = total_value
        else:
            fitness = 0
        FIT_ARR.append(fitness)
        TIME_ARR.append(time.time())
        return fitness

    def get_prob_type(self):
        """ Return the problem type.
        Returns
        -------
        self.prob_type: string
            Specifies problem type as 'discrete', 'continuous', 'tsp'
            or 'either'.
        """
        return self.prob_type

np.random.seed(111)
fitness = mlrose.CustomFitness(my_fitness)


lengths = [10,20,30,40,50,60]
max_fitness = [46,210,377,689,1166,1698]
index_plot1 = 0
index_plot2 = 5

curves_annealing =[]
curves_ga =[]
curves_rhl =[]
curves_mimic = []

times_annealing =[]
times_ga =[]
times_rhl =[]
times_mimic = []

for length  in lengths:
    weights = np.random.randint(length-1, size=length) + 1
    values =  np.random.randint(length-1, size=length) + 1
    print(weights)
    max_weight_pct = 0.6
    fitness = my_Knapsack(weights, values, max_weight_pct)

    print("length", length)

    problem = mlrose.DiscreteOpt(length=length, fitness_fn=fitness, maximize=True, max_val=2)
    time0 = time.time()
    init_state =  np.random.randint(2, size=length)

    TIME_ARR  = []
    FIT_ARR = []
    max_iters = 10000000

    decay = 0.2
    best_state, best_fitness = mlrose.simulated_annealing(problem, schedule = mlrose.GeomDecay(decay),
                                                      max_attempts = 10, max_iters = max_iters,
                                                      init_state = init_state, curve = False,
                                                                            random_state = 111)

    curve_annealing = np.copy(np.array(FIT_ARR))
    time_annealing  = np.copy(np.array(TIME_ARR))

    print("SA :", best_fitness)

    TIME_ARR = []
    FIT_ARR  = []
    max_iters = 200000
    best_state, best_fitness= mlrose.random_hill_climb(problem, restarts = 200,
                                                               max_attempts= 1000, max_iters=max_iters,
                                                               init_state=init_state, curve=False,
                                                                                random_state=111)


    print("HCA :",best_fitness)
    curve_rhl = np.copy(np.array(FIT_ARR))
    time_rhl  = np.copy(np.array(TIME_ARR))


    FIT_ARR = []
    TIME_ARR = []
    max_iters = 100
    best_state, best_fitness = mlrose.genetic_alg(problem, pop_size=3000, mutation_prob=0.01,
                                                         max_attempts=500, max_iters = max_iters, curve= False,
                                                                                random_state=111)
    curve_ga = np.copy(np.array(FIT_ARR))
    time_ga  = np.copy(np.array(TIME_ARR))
    print("GA: ",best_fitness)


    FIT_ARR =[]
    TIME_ARR =[]
    max_iters = 50
    best_state, best_fitness = mlrose.mimic(problem, pop_size=7000, keep_pct=0.1,
                                                     max_attempts=100, max_iters = max_iters, curve=False,
                                                                                random_state=111)
    curve_mimic =np.copy(np.array(FIT_ARR))
    time_mimic = np.copy(np.array(TIME_ARR))
    print("MIMIC: ",best_fitness)

    curves_annealing.append(curve_annealing)
    curves_ga.append(curve_ga)
    curves_mimic.append(curve_mimic)
    curves_rhl.append(curve_rhl)

    times_annealing.append(time_annealing)
    times_ga.append(time_ga)
    times_mimic.append(time_mimic)
    times_rhl.append(time_rhl)
curves_annealing = np.array(curves_annealing)
curves_ga = np.array(curves_ga)
curves_mimic = np.array(curves_mimic)
curves_rhl   = np.array(curves_rhl)

times_annealing = np.array(times_annealing)
times_ga = np.array(times_ga)
times_mimic = np.array(times_mimic)
times_rhl   = np.array(times_rhl)



np.save("data/ks_SA.npy", curves_annealing)
np.save("data/ks_ga.npy",  curves_ga )
np.save("data/ks_mimic.npy", curves_mimic)
np.save("data/ks_rhc",  curves_rhl )
np.save("data/ks_timesSA.npy", times_annealing)
np.save("data/ks_timesga.npy",  times_ga )
np.save("data/ks_timesmimic.npy", times_mimic)
np.save("data/ks_timesrhc",  times_rhl )


curves_annealing = np.load('data/ks_SA.npy', allow_pickle=True)
curves_ga = np.load('data/ks_ga.npy',allow_pickle=True)
curves_mimic = np.load('data/ks_mimic.npy',allow_pickle=True)
curves_rhl = np.load('data/ks_rhc.npy',allow_pickle=True)
times_annealing = np.load('data/ks_timesSA.npy',allow_pickle=True)
times_ga = np.load('data/ks_timesga.npy',allow_pickle=True)
times_mimic = np.load('data/ks_timesmimic.npy',allow_pickle=True)
times_rhl = np.load('data/ks_timesrhc.npy',allow_pickle=True)


# number of evaluation vs Function evaluation



max_curve_annealing = my_best_fitness(curves_annealing[index_plot1])
max_curve_ga        = my_best_fitness(curves_ga[index_plot1])
max_curve_mimic     = my_best_fitness(curves_mimic[index_plot1])
max_curve_rhl       = my_best_fitness(curves_rhl[index_plot1])


Y_SA = np.array(max_curve_annealing)
Y_HCA = np.array(max_curve_rhl)
Y_GA = np.array(max_curve_ga)
Y_MIMIC = np.array(max_curve_mimic)

X_SA = np.array(times_annealing[index_plot1])
X_HCA = np.array(times_rhl[index_plot1])
X_GA = np.array(times_ga[index_plot1])
X_MIMIC = np.array(times_mimic[index_plot1])

X_SA =X_SA - X_SA[0]
X_HCA = X_HCA - X_HCA[0]
X_GA = X_GA - X_GA[0]
X_MIMIC = X_MIMIC - X_MIMIC[0]

ig, ax = plt.subplots()
ax.plot(X_SA, Y_SA, color='steelblue', label="SA", linewidth=1.5)
ax.plot(X_SA[-1], Y_SA[-1], 'ro')
ax.plot(X_HCA, Y_HCA, color='green', label="HCA", linewidth=1.5, linestyle = '--')
ax.plot(X_HCA[-1], Y_HCA[-1], 'ro')
ax.plot(X_GA, Y_GA, color='red', label="GA", linewidth=1.5, linestyle ='--')
ax.plot(X_GA[-1], Y_GA[-1], 'ro')
ax.plot(X_MIMIC, Y_MIMIC, color='black', label="MIMIC", linewidth=1.5)
ax.plot(X_MIMIC[-1], Y_MIMIC[-1], 'ro')

ax.legend(loc="best")
plt.grid(False, linestyle='--')
plt.title("Best fitness vs clock time.")
plt.xlim((0,0.0015))
plt.xlabel("Clock time (seconds)")
plt.ylabel("Best Fitness function")
plt.savefig("plots/knapsack2.png")



max_curve_annealing = my_best_fitness(curves_annealing[index_plot2])
max_curve_ga        = my_best_fitness(curves_ga[index_plot2])
max_curve_mimic     = my_best_fitness(curves_mimic[index_plot2])
max_curve_rhl       = my_best_fitness(curves_rhl[index_plot2])

Y_SA = np.array(max_curve_annealing)
Y_HCA = np.array(max_curve_rhl)
Y_GA = np.array(max_curve_ga)
Y_MIMIC = np.array(max_curve_mimic)

X_SA = np.array(times_annealing[index_plot2])
X_HCA = np.array(times_rhl[index_plot2])
X_GA = np.array(times_ga[index_plot2])
X_MIMIC = np.array(times_mimic[index_plot2])

X_SA =X_SA - X_SA[0]
X_HCA = X_HCA - X_HCA[0]
X_GA = X_GA - X_GA[0]
X_MIMIC = X_MIMIC - X_MIMIC[0]

ig, ax = plt.subplots()
ax.plot(X_SA, Y_SA, color='steelblue', label="SA", linewidth=1.5)
ax.plot(X_SA[-1], Y_SA[-1], 'ro')
ax.plot(X_HCA, Y_HCA, color='green', label="HCA", linewidth=1.5, linestyle = ':')
ax.plot(X_HCA[-1], Y_HCA[-1], 'ro')
ax.plot(X_GA, Y_GA, color='red', label="GA", linewidth=1.5, linestyle ='--')
ax.plot(X_GA[-1], Y_GA[-1], 'ro')
ax.plot(X_MIMIC, Y_MIMIC, color='black', label="MIMIC", linewidth=1.5)
ax.plot(X_MIMIC[-1], Y_MIMIC[-1], 'ro')

ax.legend(loc="best")
plt.grid(False, linestyle='--')
plt.title("Best fitness vs clock time.")
plt.xlim((0,5))
plt.xlabel("Clock time (seconds)")
plt.ylabel("Best Fitness function")
plt.savefig("plots/knapsack1.png")





# Reach the plot
eval_sa = []
eval_ga = []
eval_mimic = []
eval_rhc = []
print(len(curves_annealing))
for i in range(len(lengths)):
    eval_sa.append(my_reach_max(curves_annealing[i],max_fitness[i]))
    eval_ga.append(my_reach_max(curves_ga[i],max_fitness[i]))
    eval_mimic.append(my_reach_max(curves_mimic[i], max_fitness[i]))
    eval_rhc.append(my_reach_max(curves_rhl[i], max_fitness[i]))

eval_sa =np.array(eval_sa)
eval_ga = np.array(eval_ga)
eval_mimic = np.array(eval_mimic)
eval_rhc =np.array(eval_rhc)
#ax.plot(x_arr, tr_mean, color='steelblue', marker='o', markersize=4)
ig, ax = plt.subplots()
ax.plot(lengths, eval_sa, color='steelblue', label="SA", linewidth=1.5)
ax.plot(lengths, eval_sa, color='steelblue', marker='o', markersize = 4)
ax.plot(lengths, eval_rhc, color='green', label="RHC", linewidth=1.5)
ax.plot(lengths, eval_rhc, color='green', marker='o', markersize = 4)
ax.plot(lengths, eval_ga, color='red', label="GA", linewidth=1.5)
ax.plot(lengths, eval_ga, color='red', marker='o', markersize = 4)
ax.plot(lengths, eval_mimic, color='black', label="MIMIC", linewidth=1.5)
ax.plot(lengths, eval_mimic, color='black', marker='o', markersize = 4)
ax.legend(loc="best")
plt.title("Function evaluations required to maximize the fitness.")

plt.xlabel("Bitstring length")
plt.ylabel("Function evaluation ( in thousands)")
plt.savefig("plots/knapsack3.png")

max_sa    = []
max_ga    = []
max_mimic = []
max_rhc   = []
print("len",len(curves_rhl))
for i in range(len(lengths)):
    print(i)
    max_sa.append(my_best_fitness(curves_annealing[i])[-1])
    max_ga.append(my_best_fitness(curves_ga[i])[-1])
    max_mimic.append(my_best_fitness(curves_mimic[i])[-1])
    max_rhc.append(my_best_fitness(curves_rhl[i])[-1])



ig, ax = plt.subplots()
ax.plot(lengths, max_sa, color='steelblue', label="SA", linewidth=1.5, linestyle ='-')
ax.plot(lengths, max_sa, color='steelblue', marker='o', markersize = 4, linestyle='None')
ax.plot(lengths, max_rhc, color='green', label="RHC", linewidth=1.5, linestyle ='-')
ax.plot(lengths, max_rhc, color='green', marker='*', markersize = 4, linestyle='None')
ax.plot(lengths, max_ga, color='red', label="GA", linewidth=1.5, linestyle ='-')
ax.plot(lengths, max_ga, color='red', marker='+', markersize = 8, linestyle='None')
ax.plot(lengths, max_mimic, color='black', label="MIMIC", linewidth=1.5, linestyle =':')
ax.plot(lengths, max_mimic, color='black', marker='D', markersize = 4, linestyle='None')
ax.legend(loc="best")
plt.title("Maximum fitness for different algorithms")

plt.xlabel("Bitstring length")
plt.ylabel("Maximum fitness")
plt.savefig("plots/knapsack4.png")
