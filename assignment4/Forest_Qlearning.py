# https://github.com/simoninithomas/Deep_reinforcement_learning_Course/blob/master/Q%20learning/FrozenLake/Q%20Learning%20with%20FrozenLake.ipynb
# https://stackoverflow.com/questions/14313510/how-to-calculate-moving-average-using-numpy
import math
import numpy as np
import gym
from gym.envs.toy_text.frozen_lake import generate_random_map
import matplotlib.pyplot as plt
import time
import mdptoolbox
import mdptoolbox, mdptoolbox.example
import seaborn as sns
import random

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


def Q_learning(env, total_episodes, learning_rate, max_steps, gamma, epsilon,
           max_epsilon, min_epsilon, decay_rate, verbose= True):
    qtable = np.zeros((env.nS, env.nA))
    time0         = time.time()
    clean_episode = True
    episode_length = total_episodes
    time_length    = 10e6
    for episode in range(total_episodes):
        # Reset the environment
        state = np.random.randint(env.nS, size=1)[0]

        step = 0
        done = False
        total_rewards = 0
        REWARD_ARR = []
        for step in range(max_steps):
            # 3. Choose an action a in the current world state (s)
            ## First we randomize a number
            exp_exp_tradeoff = random.uniform(0, 1)
            ## If this number > greater than epsilon --> exploitation (taking the biggest Q value for this state)
            if exp_exp_tradeoff > epsilon:
                action = np.argmax(qtable[state, :])
            # Else doing a random choice --> exploration
            else:
                action = env.random_action()
            # Take the action (a) and observe the outcome state(s') and reward (r)
            new_state, reward, done, info = env.new_state(state, action)
            if reward > 0:
                total_rewards += reward
            # Update Q(s,a):= Q(s,a) + lr [R(s,a) + gamma * max Q(s',a') - Q(s,a)]
            # qtable[new_state,:] : all the actions we can take from new state
            qtable[state, action] = qtable[state, action] + learning_rate * (
                    reward + gamma * np.max(qtable[new_state, :]) - qtable[state, action])
            # Our new state is state
            state = new_state


        # Reduce epsilon (because we need less and less exploration)
        epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-decay_rate * episode)

        if verbose:
            if math.fmod(episode,100)==0:
                print(episode, total_rewards, epsilon, decay_rate)
        rewards.append(total_rewards)

        if np.array(rewards)[-100:].mean() > 995 and clean_episode==True:
            episode_length = episode
            time_length = time.time() - time0
            clean_episode = False
            break

    return time_length, episode_length, qtable, rewards


class my_env:
    def __init__(self, n_states, n_actions):
        self.P =  [[[] for x in range(n_actions)] for y in range(n_states)]
        self.nS = n_states
        self.nA = n_actions

    def new_state(self,state,action):
        listy = env.P[state][action]
        p = []
        for item in listy:
            p.append(item[0])
        p = np.array(p)
        #print(p,state)
        chosen_index = np.random.choice(env.nS, 1, p=p)[0]
        chosen_item = listy[chosen_index]
        return chosen_item[1],chosen_item[0], chosen_item[2],chosen_item[3]

    def random_action(self):
        action = np.random.randint(2, size=1)[0]
        return action


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

size = 10
env = my_forest(size)


import random
action_size = env.nA
state_size = env.nS
qtable = np.zeros((state_size, action_size))

total_episodes = 10000
learning_rate = 0.1         # Learning rate
max_steps = 1000           # Max steps per episode
gamma = 0.9          # Discounting rate

# Exploration parameters

epsilon = 0.1                 # Exploration rate
max_epsilon = 1.0             # Exploration probability at start
min_epsilon = 10e-9           # Minimum exploration probability
decay_rate = 0.001       # Exponential decay rate for exploration prob

# List of rewards
rewards = []

def moving_average(a, n=3) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

time00 = time.time()
time_length, episode_length, qtable,rewards = Q_learning(env, total_episodes, learning_rate, max_steps, gamma, epsilon,
                                                  max_epsilon, min_epsilon, decay_rate, verbose = True)
time10 = time.time()

qbest = np.empty(env.nS)
#print(qtable)
for state in range(env.nS):
    qbest[state] = np.argmax(qtable[state,:])
print("Score over time: " + str(sum(rewards) / total_episodes))
#print(qtable)
fig, ax = plt.subplots()
rewards_moving1 = moving_average(rewards,10)
X1 = np.arange(1,len(rewards_moving1)+1,1)



action_size = env.nA
state_size = env.nS
qtable = np.zeros((state_size, action_size))

total_episodes = 10000
learning_rate = 0.1         # Learning rate
max_steps = 1000           # Max steps per episode
gamma = 0.9          # Discounting rate

# Exploration parameters

epsilon = 0.1                 # Exploration rate
max_epsilon = 1.0             # Exploration probability at start
min_epsilon = 10e-9           # Minimum exploration probability
decay_rate = 0.01       # Exponential decay rate for exploration prob

# List of rewards
rewards = []

def moving_average(a, n=3) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

time10 = time.time()
time_length, episode_length, qtable,rewards = Q_learning(env, total_episodes, learning_rate, max_steps, gamma, epsilon,
                                                  max_epsilon, min_epsilon, decay_rate, verbose = True)
time11 = time.time()
qbest = np.empty(env.nS)
#print(qtable)
for state in range(env.nS):
    qbest[state] = np.argmax(qtable[state,:])
print("Score over time: " + str(sum(rewards) / total_episodes))
#print(qtable)
fig, ax = plt.subplots()
rewards_moving2 = moving_average(rewards,10)
X2 = np.arange(1,len(rewards_moving2)+1,1)





action_size = env.nA
state_size = env.nS
qtable = np.zeros((state_size, action_size))

total_episodes = 10000
learning_rate = 0.1         # Learning rate
max_steps = 1000           # Max steps per episode
gamma = 0.9          # Discounting rate

# Exploration parameters

epsilon = 0.1                 # Exploration rate
max_epsilon = 1.0             # Exploration probability at start
min_epsilon = 10e-9           # Minimum exploration probability
decay_rate = 0.05       # Exponential decay rate for exploration prob

# List of rewards
rewards = []

def moving_average(a, n=3) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

time20 = time.time()
time_length, episode_length, qtable,rewards = Q_learning(env, total_episodes, learning_rate, max_steps, gamma, epsilon,
                                                  max_epsilon, min_epsilon, decay_rate, verbose = True)
time21 = time.time()
qbest = np.empty(env.nS)
#print(qtable)
for state in range(env.nS):
    qbest[state] = np.argmax(qtable[state,:])
print("Score over time: " + str(sum(rewards) / total_episodes))
#print(qtable)
fig, ax = plt.subplots()
rewards_moving3 = moving_average(rewards,10)
X3 = np.arange(1,len(rewards_moving3)+1,1)

ax.plot(X1, rewards_moving1, color='steelblue', label="Mean Reward (decay rate = 0.001)", linewidth=2.0, linestyle='-')
ax.plot(X2, rewards_moving2, color='red', label="Mean Reward (decay rate = 0.01)", linewidth=2.0, linestyle='-')
ax.plot(X3, rewards_moving3, color='black', label="Mean Reward (decay rate = 0.05)", linewidth=2.0, linestyle='-')
ax.legend(loc='best', frameon=True)
plt.grid(False, linestyle='--')
plt.title('Mean Reward vs number of episodes, Q learning (grid size = 10).')
plt.ylabel('Mean Reward (over 10 episodes')
plt.xlabel(' Number of episodes')
plt.savefig('plots/forest_Qlearning_curve.png')

print("eps = 0.01, num_episodes: ", time10-time00, len(X1))
print("eps = 0.1, num_episodes: ", time11-time10, len(X2))
print("eps = 0.5, num_episodes: ", time21-time20, len(X3))

action_size = env.nA
state_size = env.nS
qtable = np.zeros((state_size, action_size))

total_episodes = 10000
learning_rate = 0.1         # Learning rate
max_steps = 1000           # Max steps per episode
gamma = 0.9          # Discounting rate

# Exploration parameters

epsilon = 0.8                  # Exploration rate
max_epsilon = 0.8              # Exploration probability at start
min_epsilon = 0.8              # Minimum exploration probability
decay_rate = 1.0               # Exponential decay rate for exploration prob

# List of rewards
rewards = []

def moving_average(a, n=3) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

time_length, episode_length, qtable,rewards = Q_learning(env, total_episodes, learning_rate, max_steps, gamma, epsilon,
                                                  max_epsilon, min_epsilon, decay_rate, verbose = True)

qbest = np.empty(env.nS)
#print(qtable)
for state in range(env.nS):
    qbest[state] = np.argmax(qtable[state,:])
print("Score over time: " + str(sum(rewards) / total_episodes))
#print(qtable)
fig, ax = plt.subplots()
rewards_moving1 = moving_average(rewards,10)
X1 = np.arange(1,len(rewards_moving1)+1,1)


print(qtable)







action_size = env.nA
state_size = env.nS
qtable = np.zeros((state_size, action_size))

total_episodes = 5000
learning_rate = 0.1         # Learning rate
max_steps = 1000           # Max steps per episode
gamma = 0.9          # Discounting rate

# Exploration parameters

epsilon = 0.02                  # Exploration rate
max_epsilon = 0.02              # Exploration probability at start
min_epsilon = 0.02              # Minimum exploration probability
decay_rate = 1.0               # Exponential decay rate for exploration prob

# List of rewards
rewards = []

def moving_average(a, n=3) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

time_length, episode_length, qtable,rewards = Q_learning(env, total_episodes, learning_rate, max_steps, gamma, epsilon,
                                                  max_epsilon, min_epsilon, decay_rate, verbose = True)

qbest = np.empty(env.nS)
#print(qtable)
for state in range(env.nS):
    qbest[state] = np.argmax(qtable[state,:])
print("Score over time: " + str(sum(rewards) / total_episodes))
#print(qtable)
fig, ax = plt.subplots()
rewards_moving2 = moving_average(rewards,10)
X2 = np.arange(1,len(rewards_moving2)+1,1)

ax.plot(X1, rewards_moving1, color='steelblue', label="Mean Reward (epsilon = 0.8)", linewidth=2.0, linestyle='-')
ax.plot(X2, rewards_moving2, color='red', label="Mean Reward (epsilon = 0.05)", linewidth=2.0, linestyle='-')
ax.legend(loc='best', frameon=True)
plt.grid(False, linestyle='--')
plt.title('Mean Reward vs number of episodes, Q learning (grid size = 10).')
plt.ylabel('Mean Reward (over 10 episodes')
plt.xlabel(' Number of episodes')
plt.savefig('plots/forest_Qlearning_curve_eps08.png')


print(qtable)