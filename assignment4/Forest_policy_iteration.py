#https://github.com/dennybritz/reinforcement-learning/blob/master/DP/Policy%20Iteration.ipynb
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

def policy_eval(policy, env, v_prev,  discount_factor=1.0, theta=0.0001 ):

    """
    Evaluate a policy given an environment and a full description of the environment's dynamics.
    Args:
        policy: [S, A] shaped matrix representing the policy.
        env: OpenAI env. env.P represents the transition probabilities of the environment.
            env.P[s][a] is a list of transition tuples (prob, next_state, reward, done).
            env.nS is a number of states in the environment.
            env.nA is a number of actions in the environment.
        theta: We stop evaluation once our value function change is less than theta for all states.
        discount_factor: Gamma discount factor.

    Returns:
        Vector of length env.nS representing the value function.
    """
    # Start with a random (all 0) value function

    V = v_prev
    num_iters = 0
    while True:
        num_iters = num_iters + 1
        delta = 0
        # For each state, perform a "full backup"
        for s in range(env.nS):
            v = 0
            # Look at the possible next actions
            for a, action_prob in enumerate(policy[s]):
                # For each action, look at the possible next states...
                for prob, next_state, reward, done in env.P[s][a]:
                    # Calculate the expected value
                    v += action_prob * prob * (reward + discount_factor * V[next_state])
            # How much our value function changed (across any states)
            delta = max(delta, np.abs(v - V[s]))
            V[s] = v
        # Stop evaluating once our value function change is below a threshold
        if delta < theta:
            break
    return np.array(V)

def policy_improvement(env, policy_eval_fn=policy_eval, discount_factor=1.0):
    """
    Policy Improvement Algorithm. Iteratively evaluates and improves a policy
    until an optimal policy is found.

    Args:
        env: The OpenAI environment.
        policy_eval_fn: Policy Evaluation function that takes 3 arguments:
            policy, env, discount_factor.
        discount_factor: gamma discount factor.

    Returns:
        A tuple (policy, V).
        policy is the optimal policy, a matrix of shape [S, A] where each state s
        contains a valid probability distribution over actions.
        V is the value function for the optimal policy.

    """

    def one_step_lookahead(state, V):
        """
        Helper function to calculate the value for all action in a given state.

        Args:
            state: The state to consider (int)
            V: The value to use as an estimator, Vector of length env.nS

        Returns:
            A vector of length env.nA containing the expected value of each action.
        """
        A = np.zeros(env.nA)
        for a in range(env.nA):
            for prob, next_state, reward, done in env.P[state][a]:
                A[a] += prob * (reward + discount_factor * V[next_state])
        return A

    # Start with a random policy
    policy = np.ones([env.nS, env.nA]) / env.nA
    V_ARR = []
    V_SUM = []
    V = np.zeros(env.nS)

    while True:
        print("iter)")
        # Evaluate the current policy
        #print("A", time.time())
        V = policy_eval_fn(policy, env, V, discount_factor = discount_factor)

        # Will be set to false if we make any changes to the policy
        policy_stable = True
        #print("B", time.time())
        # For each state...
        for s in range(env.nS):
            # The best action we would take under the current policy
            chosen_a = np.argmax(policy[s])
            # Find the best action by one-step lookahead
            # Ties are resolved arbitarily
            action_values = one_step_lookahead(s, V)
            best_a = np.argmax(action_values)
            # Greedily update the policy
            if chosen_a != best_a:
                policy_stable = False
            policy[s] = np.eye(env.nA)[best_a]

        #print("C)", time.time())
        # If the policy is stable we've found an optimal policy. Return it
        V_ARR.append(V)
        V_SUM.append(V.sum())
        if policy_stable:
            return policy, V_ARR, V_SUM


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

np.random.seed(1111)

#
# Different sizes
#
N_ITERS  = []
SIZE     = np.arange(3,15,1)
TIME_ARR = []
for size in SIZE:
    print(size)
    np.random.seed(1111)
    env = my_forest(size)
    print("nStates  ", env.nS)
    time0 = time.time()
    policy, V_ARR, V_SUM = policy_improvement(env, discount_factor= 0.99)
    time1 = time.time()
    N_ITERS.append(len(V_ARR))
    TIME_ARR.append(time1 - time0)
"""
fig, ax = plt.subplots()
ax.plot(SIZE, N_ITERS , color='red', label="Number of iterations", linewidth=2.0, linestyle='-')
ax.plot(SIZE, N_ITERS , color='red',  marker='o', markersize = 2)

ax.legend(loc='best', frameon=True)
plt.grid(False, linestyle='--')
plt.title('Number of iteration to converge for different problem size \n (Policy iteration)')
plt.ylabel('Number of iterations')
plt.xlabel('Size of the problem')
plt.tight_layout()
plt.savefig('plots/lake_pi_size.png')
"""

fig, ax = plt.subplots()
ax.plot(SIZE, TIME_ARR , color='red', label="Clock time", linewidth=2.0, linestyle='-')
ax.plot(SIZE, TIME_ARR , color='red',  marker='o', markersize = 2)

ax.legend(loc='best', frameon=True)
plt.grid(False, linestyle='--')
plt.title('Time necessary to converge \n (policy iteration)')
plt.ylabel('Clock time')
plt.xlabel('Size of the problem')
plt.tight_layout()
plt.savefig('plots/lake_pi_Time.png')









size = 10
env = my_forest(size)


policy99,V_ARR99, V_SUM99 = policy_improvement(env,  discount_factor = 0.99)
policy9,V_ARR9, V_SUM9 = policy_improvement(env,    discount_factor = 0.9)
policy7, V_ARR7, V_SUM7 = policy_improvement(env,  discount_factor =0.7)



X99 = np.arange(1,len(V_SUM99)+1,1)
X9 = np.arange(1,len(V_SUM9)+1,1)
X7 = np.arange(1,len(V_SUM7)+1,1)
fig, ax = plt.subplots()
ax.plot(X99, V_SUM99 , color='steelblue', label="gamma = 0.99", linewidth=2.0, linestyle='-')
ax.plot(X9, V_SUM9 , color='red', label="gamma = 0.9 ", linewidth=2.0, linestyle='-')
ax.plot(X7, V_SUM7 , color='purple', label="gamma = 0.7", linewidth=2.0, linestyle='-')
ax.legend(loc='best', frameon=True)
plt.grid(False, linestyle='--')
plt.title('Sum of V values vs Number of iterations, size = 10 (Policy iteration). ')
plt.ylabel('Sum of V values')
plt.xlabel('Iterations')
plt.tight_layout()
plt.savefig('plots/forest_pi_plot.png')

draw_policy(policy99, V_ARR99[len(V_ARR99)-1], env, size, "plots/forest_policy_iteration.png", "Policy visualization \n(policy iteration), size = 10")