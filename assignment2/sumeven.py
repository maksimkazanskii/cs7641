import mlrose
import numpy as np
import myplots
import matplotlib.pyplot as plt
import time
import math
# https://mlrose.readthedocs.io/en/stable/source/tutorial1.html

# Define alternative N-Queens fitness function for maximization problem

def my_fitness(state):

   fitness = 0
   count = np.sum(state)
   if math.fmod(count,2)==0:
        fitness =  fitness+1
   fitness = fitness + count
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

fitness = mlrose.CustomFitness(my_fitness)

lengths = [10, 20, 30, 40, 50, 60, 70, 80]
max_fitness = [11, 21, 31, 41, 51, 61, 71, 81] # Should be manually written
index_plot = 5 # What type

curves_annealing =[]
curves_ga =[]
curves_rhl =[]
curves_mimic = []

times_annealing =[]
times_ga =[]
times_rhl =[]
times_mimic = []

for length  in lengths:
    print("length", length)
    problem = mlrose.DiscreteOpt(length=length, fitness_fn=fitness, maximize=True, max_val=2)
    time0 = time.time()
    init_state =  np.random.randint(2, size=length)

    TIME_ARR  = []
    FIT_ARR = []
    max_iters = 10000
    best_state, best_fitness = mlrose.simulated_annealing(problem, schedule = mlrose.ExpDecay(),
                                                      max_attempts = 500, max_iters = max_iters,
                                                      init_state = init_state, curve = False,
                                                                            random_state = 111)

    curve_annealing = np.copy(np.array(FIT_ARR))
    time_annealing  = np.copy(np.array(TIME_ARR))


    print("SA :", best_fitness)

    TIME_ARR = []
    FIT_ARR  = []
    max_iters = 5000
    best_state, best_fitness= mlrose.random_hill_climb(problem, restarts =100,
                                                               max_attempts=500, max_iters=max_iters,
                                                               init_state=init_state, curve=False,
                                                                                random_state=111)


    print("HCA :",best_fitness)
    curve_rhl = np.copy(np.array(FIT_ARR))
    time_rhl  = np.copy(np.array(TIME_ARR))


    FIT_ARR = []
    TIME_ARR = []
    max_iters = 300
    best_state, best_fitness = mlrose.genetic_alg(problem, pop_size=200, mutation_prob=0.05,
                                                         max_attempts=500, max_iters = max_iters, curve= False,
                                                                                random_state=111)
    curve_ga = np.copy(np.array(FIT_ARR))
    time_ga  = np.copy(np.array(TIME_ARR))
    print("GA: ",best_fitness)


    FIT_ARR =[]
    TIME_ARR =[]
    max_iters = 50
    best_state, best_fitness = mlrose.mimic(problem, pop_size=500, keep_pct=0.2,
                                                     max_attempts=10, max_iters = max_iters, curve=False,
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





# number of evaluation vs Function evaluation


max_curve_annealing = my_best_fitness(curves_annealing[index_plot])
max_curve_ga        = my_best_fitness(curves_ga[index_plot])
max_curve_mimic     = my_best_fitness(curves_mimic[index_plot])
max_curve_rhl       = my_best_fitness(curves_rhl[index_plot])


Y_SA = np.array(max_curve_annealing)
Y_HCA = np.array(max_curve_rhl)
Y_GA = np.array(max_curve_ga)
Y_MIMIC = np.array(max_curve_mimic)

max_value = 800
X_ARR  =np.arange(1,max_value+1,1)

ig, ax = plt.subplots()
ax.plot(X_ARR, Y_SA[:max_value], color='steelblue', label="SA", linewidth=1.5)
ax.plot(X_ARR, Y_HCA[:max_value], color='green', label="HCA", linewidth=1.5)
ax.plot(X_ARR, Y_GA[:max_value], color='red', label="GA", linewidth=1.5)
ax.plot(X_ARR, Y_MIMIC[:max_value], color='black', label="MIMIC", linewidth=1.5)
ax.legend(loc="best")
plt.title("Best fitness vs function evaluation.")

plt.xlabel("Function avaluations")
plt.ylabel("Best Fitness function")
plt.savefig("plots/sumeven1.png")


# number of evaluation vs Time

max_curve_annealing = my_best_fitness(curves_annealing[index_plot])
max_curve_ga        = my_best_fitness(curves_ga[index_plot])
max_curve_mimic     = my_best_fitness(curves_mimic[index_plot])
max_curve_rhl       = my_best_fitness(curves_rhl[index_plot])


X_SA = np.array(times_annealing[index_plot])
X_HCA = np.array(times_rhl[index_plot])
X_GA = np.array(times_ga[index_plot])
X_MIMIC = np.array(times_mimic[index_plot])

X_SA =X_SA - X_SA[0]
X_HCA = X_HCA - X_HCA[0]
X_GA = X_GA - X_GA[0]
X_MIMIC = X_MIMIC - X_MIMIC[0]

ig, ax = plt.subplots()
ax.plot(X_SA, Y_SA, color='steelblue', label="SA", linewidth=1.5)
ax.plot(X_HCA, Y_HCA, color='green', label="HCA", linewidth=1.5)
ax.plot(X_GA, Y_GA, color='red', label="GA", linewidth=1.5)
ax.plot(X_MIMIC, Y_MIMIC, color='black', label="MIMIC", linewidth=1.5)

ax.legend(loc="best")
plt.grid(False, linestyle='--')
plt.title("Best fitness vs clock time.")

plt.xlabel("Clock time (seconds)")
plt.ylabel("Best Fitness function")
plt.savefig("plots/sumeven3.png")



# Reach the plot
eval_sa = []
eval_ga = []
eval_mimic = []
eval_rhc = []
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
plt.savefig("plots/sumeven2.png")