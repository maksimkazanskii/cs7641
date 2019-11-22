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
    plt.tight_layout()
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


np.random.seed(1111)

#
# Different sizes
#

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
    policy, V_ARR, V_SUM  = policy_improvement(env, discount_factor = 0.99)
    time1 = time.time()
    N_ITERS.append(len(V_SUM))
    TIME_ARR.append(time1 - time0)

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

np.random.seed(1111)
size = 15
random_map = generate_random_map(size=size, p=0.85)
print(random_map)
env = gym.make("FrozenLake-v0", desc=random_map, is_slippery = False)
env.reset()
env.render()

policy, V_ARR, V_SUM = policy_improvement(env,  discount_factor = 0.99)


np.random.seed(1111)
size1 = 30
random_map = generate_random_map(size=size1, p=0.85)
print(random_map)
env1 = gym.make("FrozenLake-v0", desc=random_map, is_slippery = False)
env1.reset()
env1.render()
policy1, V_ARR1,V_SUM1 = policy_improvement(env1,  discount_factor = 0.99)
print(V_SUM1)
plt.tight_layout()
X = np.arange(1,len(V_SUM1)+1,1)
fig, ax = plt.subplots()
ax.plot(X, V_SUM1 , color='steelblue', label="Sum of V Values", linewidth=2.0, linestyle='-')
ax.plot(X, V_SUM1 , color='steelblue',  marker='o', markersize = 4)

ax.legend(loc='best', frameon=True)
plt.grid(False, linestyle='--')
plt.title('Sum of V values vs Number of iterations, \n( policy-iteration algorithm ). ')
plt.ylabel('Sum of V Values')
plt.xlabel('Iterations')
plt.tight_layout()
plt.savefig('plots/lake_policy_iteration_plot.png')

draw_policy(policy1, V_ARR1[len(V_ARR1)-1], env1, size1, "plots/lake_policy_iteration2.png", "Policy visualization (policy iteration), size = 30")
draw_policy(policy, V_ARR[len(V_ARR)-1], env, size, "plots/lake_policy_iteration1.png", "Policy visualization (policy iteration), size = 15")