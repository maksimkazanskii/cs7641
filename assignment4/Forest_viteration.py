# https://github.com/dennybritz/reinforcement-learning/blob/master/DP/Policy%20Iteration.ipynb
import numpy as np
import gym
from gym.envs.toy_text.frozen_lake import generate_random_map
import matplotlib.pyplot as plt
import time
import mdptoolbox
import mdptoolbox, mdptoolbox.example
import seaborn as sns


def draw_policy(policy, V, env, size, file_name, plt_title):
    import matplotlib.pyplot as plt


    #desc = env.desc.astype('U')

    policy_flat = np.argmax(policy, axis=1)
    V           = policy_flat
    policy_grid = np.copy(policy_flat)
    sns.set()

    policy_list = np.chararray((size), unicode=True)

    policy_list[np.where(policy_grid == 0)] = 'Wait'
    policy_list[np.where(policy_grid == 1)] = 'Cut'

    a4_dims = (3, 9)


    fig, ax = plt.subplots(figsize = a4_dims)

    V= V.reshape((size,1))
    policy_list = policy_list.reshape((size,1))
    print(V.shape,policy_list.shape)


    sns.heatmap(V, annot=policy_list, fmt='', ax=ax)
    plt.title(plt_title)
    plt.tight_layout()
    plt.savefig(file_name)
    plt.close()

    return True

def value_iteration(env, theta=10e-8, discount_factor=1.0):

    def one_step_lookahead(state, V):
        A = np.zeros(env.nA)
        for a in range(env.nA):
            for prob, next_state, reward, done in env.P[state][a]:
                A[a] += prob * (reward + discount_factor * V[next_state])
        return A
    V = np.zeros(env.nS)
    DELTA_ARR = []
    V_ARR = []
    V_SUM = []
    while True:
        delta = 0
        for s in range(env.nS):
            # Do a one-step lookahead to find the best action
            A = one_step_lookahead(s, V)
            best_action_value = np.max(A)
            # Calculate delta across all states seen so farg
            delta = max(delta, np.abs(best_action_value - V[s]))
            # Update the value function. Ref: Sutton book eq. 4.10.
            V[s] = best_action_value
            # Check if we can stop
        DELTA_ARR.append(delta)
        V_ARR.append(V)
        V_SUM.append(V.sum())
        if delta < theta:
            break
    # Create a deterministic policy using the optimal value function
    policy = np.zeros([env.nS, env.nA])
    for s in range(env.nS):
        # One step lookahead to find the best action for this state
        A = one_step_lookahead(s, V)
        best_action = np.argmax(A)
        # Always take the best action
        policy[s, best_action] = 1.0
    return DELTA_ARR, V_ARR,V_SUM, policy

class my_env:
    def __init__(self, n_states, n_actions):
        self.P =  [[[] for x in range(n_actions)] for y in range(n_states)]
        self.nS = n_states
        self.nA = n_actions

def my_forest(size):
    np.random.seed(1111)
    n_states  = size
    n_actions = 2
    P, R = mdptoolbox.example.forest(S=n_states, r1=4,r2=50, p=0.6)

    env = my_env(n_states, n_actions)
    for action in range(0, n_actions):
        for state in range(0, n_states):
            for state_slash in range(0,n_states):
                reward = R[state][action]
                env.P[state][action].append([P[action][state][state_slash], state_slash, reward, False])
    return env

N_ITERS  = []
SIZE     = np.arange(3,15,1)
TIME_ARR = []
for size in SIZE:
    print(size)
    np.random.seed(1111)
    env = my_forest(size)
    print("nStates  ", env.nS)
    time0 = time.time()
    DELTA_ARR, V_ARR, V_SUM, policy = value_iteration(env, 10e-10, 0.99)
    time1 = time.time()
    N_ITERS.append(len(V_ARR))
    TIME_ARR.append(time1 - time0)
"""
fig, ax = plt.subplots()
ax.plot(SIZE, N_ITERS , color='red', label="Number of iterations", linewidth=2.0, linestyle='-')
ax.plot(SIZE, N_ITERS , color='red',  marker='o', markersize = 4)

ax.legend(loc='best', frameon=True)
plt.grid(False, linestyle='--')
plt.title('Number of iterations to converge \n (Value iteration)')
plt.ylabel('Number of iterations')
plt.xlabel('Size of the problem.')
plt.tight_layout()
plt.savefig('plots/forest_vi_iters.png')
"""
fig, ax = plt.subplots()
ax.plot(SIZE, TIME_ARR , color='red', label="Clock time", linewidth=2.0, linestyle='-')
ax.plot(SIZE, TIME_ARR , color='red',  marker='o', markersize = 4)

ax.legend(loc='best', frameon=True)
plt.grid(False, linestyle='--')

plt.title('Clock time necessary to converge \n (Value iteration)')
plt.ylabel('Clock time')
plt.xlabel('Size of the problem.')
plt.tight_layout()
plt.savefig('plots/forest_vi_time.png')




size = 10
env = my_forest(size)


DELTA_ARR99, V_ARR99, V_SUM99, policy99 = value_iteration(env,  10e-15,  0.99)
DELTA_ARR9, V_ARR9, V_SUM9, policy9 = value_iteration(env,  10e-15,  0.9)
DELTA_ARR7, V_ARR7, V_SUM7, policy7 = value_iteration(env,  10e-15,  0.7)



X99 = np.arange(1,len(V_SUM99)+1,1)
X9 = np.arange(1,len(V_SUM9)+1,1)
X7 = np.arange(1,len(V_SUM7)+1,1)
fig, ax = plt.subplots()
ax.plot(X99, V_SUM99 , color='steelblue', label="gamma = 0.99", linewidth=2.0, linestyle='-')
ax.plot(X9, V_SUM9 , color='red', label=" gamma = 0.9 ", linewidth=2.0, linestyle='-')
ax.plot(X7, V_SUM7 , color='purple', label=" gamma = 0.7", linewidth=2.0, linestyle='-')
ax.legend(loc='best', frameon=True)
plt.grid(False, linestyle='--')
plt.title('Sum of V vlaues vs Number of iterations, size = 10 (Value iteration). ')
plt.ylabel('Sum of V Values')
plt.xlabel('Iterations')
plt.xlim((0,500))
plt.tight_layout()
plt.savefig('plots/forest_vi_plot.png')

draw_policy(policy99, V_ARR99[len(V_ARR99)-1], env, size, "plots/forest_value_iteration.png", "Policy visualization \n(value iteration), size = 10")
