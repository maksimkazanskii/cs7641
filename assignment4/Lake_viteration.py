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
    print(policy)
    print(env.desc)
    desc = env.desc.astype('U')

    policy_flat = np.argmax(policy, axis=1)
    policy_grid = policy_flat.reshape((size,size))
    V           = V.reshape((size,size))
    sns.set()

    print(size)
    policy_list = np.chararray((size,size), unicode=True)

    policy_list[np.where(policy_grid == 1)] = 'v'
    policy_list[np.where(policy_grid == 2)] = '>'
    policy_list[np.where(policy_grid == 0)] = '<'
    policy_list[np.where(policy_grid == 3)] = '^'

    policy_list[np.where(desc == 'H')]  = '0'
    policy_list[np.where(desc == 'G')] = 'G'
    policy_list[np.where(desc == 'S')] = 'S'
    a4_dims = (12, 12)


    fig, ax = plt.subplots(figsize = a4_dims)
    sns.heatmap(V, annot=policy_list, fmt='', ax=ax)
    #sns_plot.figure.savefig("plots/maze_solution.png")
    plt.title(plt_title)
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
    return DELTA_ARR, V_ARR, policy

np.random.seed(1111)

import matplotlib.pyplot as plt

print("Different size of the problem")
N_ITERS  = []
SIZE     = np.arange(10,40,1)
TIME_ARR = []
for size in SIZE:
    print(size)
    np.random.seed(1111)
    random_map = generate_random_map(size=size, p=0.85)
    env = gym.make("FrozenLake-v0", desc=random_map, is_slippery = False)
    env.reset()
    time0 = time.time()
    DELTA_ARR, V_ARR, policy = value_iteration(env, 10e-10, 0.99)
    time1 = time.time()
    N_ITERS.append(len(V_ARR))
    TIME_ARR.append(time1 - time0)

fig, ax = plt.subplots()
ax.plot(SIZE, N_ITERS , color='red', label="Number of iterations", linewidth=2.0, linestyle='-')
ax.plot(SIZE, N_ITERS , color='red',  marker='o', markersize = 4)

ax.legend(loc='best', frameon=True)
plt.grid(False, linestyle='--')
plt.title('Number of iterations to converge \n (Value iteration)')
plt.ylabel('Number of iterations')
plt.xlabel('Size of the problem.')
plt.tight_layout()
plt.savefig('plots/lake_vi_iters.png')

fig, ax = plt.subplots()
ax.plot(SIZE, TIME_ARR , color='red', label="Clock time", linewidth=2.0, linestyle='-')
ax.plot(SIZE, TIME_ARR , color='red',  marker='o', markersize = 4)

ax.legend(loc='best', frameon=True)
plt.grid(False, linestyle='--')

plt.title('Clock time necessary to converge \n (Value iteration)')
plt.ylabel('Clock time')
plt.xlabel('Size of the problem.')
plt.tight_layout()
plt.savefig('plots/lake_vi_time.png')

#
# Specific size of the problem
#

np.random.seed(1111)
size1 = 15
#size = 50
random_map = generate_random_map(size=size1, p=0.85)
print(random_map)
env1 = gym.make("FrozenLake-v0", desc=random_map, is_slippery = False)
env1.reset()
env1.render()

DELTA_ARR1, V_ARR1, policy1 = value_iteration(env1,  10e-10,  0.99)

size = 30

random_map = generate_random_map(size=size, p=0.85)
print(random_map)
env = gym.make("FrozenLake-v0", desc=random_map, is_slippery = False)
env.reset()
env.render()

DELTA_ARR, V_ARR, policy = value_iteration(env,  10e-10,  0.99)


X = np.arange(1,len(DELTA_ARR)+1,1)
fig, ax = plt.subplots()
ax.plot(X, DELTA_ARR , color='steelblue', label="Delta", linewidth=2.0, linestyle='-')
ax.plot(X, DELTA_ARR , color='steelblue',  marker='o', markersize = 4)

ax.legend(loc='best', frameon=True)
plt.grid(False, linestyle='--')
plt.title('Delta vs Number of iterations, size = 30 (Value iteration). ')
plt.ylabel('Delta')
plt.xlabel('Iterations')
plt.tight_layout()
plt.savefig('plots/lake_vi_plot.png')
print(policy.shape)

print(V_ARR[len(V_ARR)-1])

draw_policy(policy, V_ARR[len(V_ARR)-1], env, size, "plots/policy2_viteration.png", "Policy visualization ( Value iteration), size =30.")
draw_policy(policy1, V_ARR1[len(V_ARR1)-1], env1, size1, "plots/policy1_viteration.png", "Policy visualization ( Value iteration), size = 15.")