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
    desc = env.desc.astype('U')
    policy_flat = np.argmax(policy, axis=1)
    policy_grid = policy_flat.reshape((size,size))
    V           = V.reshape((size,size))
    sns.set()
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

def Q_learning(env, total_episodes, learning_rate, max_steps, gamma, epsilon,
           max_epsilon, min_epsilon, decay_rate, verbose= True):
    time0         = time.time()
    clean_episode = True
    episode_length = total_episodes
    time_length     =10e6
    for episode in range(total_episodes):
        # Reset the environment
        env.seed(111)
        state = env.reset()
        env.action_space.seed(111)
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
                action = env.action_space.sample()
            # Take the action (a) and observe the outcome state(s') and reward (r)
            new_state, reward, done, info = env.step(action)
            if reward > 0:
                total_rewards += reward
            else:
                if done == True:
                    reward = -0.0001

            # Update Q(s,a):= Q(s,a) + lr [R(s,a) + gamma * max Q(s',a') - Q(s,a)]
            # qtable[new_state,:] : all the actions we can take from new state
            qtable[state, action] = qtable[state, action] + learning_rate * (
                    reward + gamma * np.max(qtable[new_state, :]) - qtable[state, action])
            total_rewards += reward
            # Our new state is state
            state = new_state
            # If done (if we're dead) :  episode
            if done == True:
                break
        # Reduce epsilon (because we need less and less exploration)
        epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-decay_rate * episode)
        if verbose:
            if math.fmod(episode,1000)==0:
                print(episode, total_rewards, epsilon, decay_rate)
        rewards.append(total_rewards)
        if np.array(rewards)[-100:].mean() > 1.98 and clean_episode==True:
            episode_length = episode
            time_length = time.time() - time0
            clean_episode = False
            break
    return time_length, episode_length, qtable, rewards



np.random.seed(111)
size = 15
random_map = generate_random_map(size=size, p=0.85)
env = gym.make("FrozenLake-v0", desc=random_map, is_slippery = False)
RANDOM_SEED = 111

env.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)
env.reset()
env.render()
env.action_space.seed(111)

import random
action_size = env.action_space.n
state_size = env.observation_space.n
qtable = np.zeros((state_size, action_size))

total_episodes = 11000
learning_rate = 0.7         # Learning rate
max_steps = 100000           # Max steps per episode
gamma = 0.7          # Discounting rate

# Exploration parameters

epsilon = 0.1                 # Exploration rate
max_epsilon = 1.0             # Exploration probability at start
min_epsilon = 10e-9           # Minimum exploration probability
decay_rate = 0.001        # Exponential decay rate for exploration prob

# List of rewards
rewards = []

def moving_average(a, n=3) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

"""
time_length, episode_length, qtable,rewards = Q_learning(env, total_episodes, learning_rate, max_steps, gamma, epsilon,
                                                  max_epsilon, min_epsilon, decay_rate, verbose = True)
print("Number of episodes (size = 15):",episode_length)
print("Time (size = 15)", time_length)
np.save("data/qtable15.npy",  qtable)
np.save("data/rewards15.npy",  rewards)
"""
qtable  = np.load('data/qtable15.npy')
rewards = np.load('data/rewards15.npy')

qbest = np.empty(env.nS)
#print(qtable)

for state in range(env.nS):

    qbest[state] = np.argmax(qtable[state,:])


print("Score over time: " + str(sum(rewards) / total_episodes))
#print(qtable)
fig, ax = plt.subplots()
rewards_moving = moving_average(rewards,100)
X = np.arange(1,len(rewards_moving)+1,1)

ax.plot(X, rewards_moving, color='steelblue', label="Mean Reward", linewidth=2.0, linestyle='-')
ax.legend(loc='best', frameon=True)
plt.grid(False, linestyle='--')
plt.title('Mean Reward vs number of episodes, Q learning (grid size = 15).')
plt.xlim((0,10000))
plt.ylabel('Mean Reward')
plt.xlabel(' Number of episodes')
plt.savefig('plots/lake_Qlearning_curve.png')
for episode in range(5):
    state = env.reset()
    step = 0
    done = False
    #print("****************************************************")
    #print("EPISODE ", episode)

    for step in range(max_steps):

        # Take the action (index) that have the maximum expected future reward given that state
        action = np.argmax(qtable[state, :])
        new_state, reward, done, info = env.step(action)
        if done:
            # Here, we decide to only print the last state (to see if our agent is on the goal or fall into an hole)
            #env.render()

            # We print the number of step it took.
            #print("Number of steps", step)
            break
        state = new_state

#draw_policy(qtable, qbest, env, size, "plots/lake_Qvalue1.png",
#                "Policy visualization (Q learning), size = 15 ")
env.close




# Large grid size



np.random.seed(111)
size = 30
random_map = generate_random_map(size=size, p=0.85)
env = gym.make("FrozenLake-v0", desc=random_map, is_slippery = False)
RANDOM_SEED = 111

env.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)
env.reset()
env.render()
env.action_space.seed(111)

import random
action_size = env.action_space.n
state_size = env.observation_space.n
qtable = np.zeros((state_size, action_size))

total_episodes = 110000
learning_rate = 0.7         # Learning rate
max_steps = 100000           # Max steps per episode
gamma = 0.6          # Discounting rate

# Exploration parameters

epsilon = 0.6                 # Exploration rate
max_epsilon = 1.0             # Exploration probability at start
min_epsilon = 10e-9           # Minimum exploration probability
decay_rate = 0.0001        # Exponential decay rate for exploration prob

# List of rewards
rewards = []

def moving_average(a, n=3) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

"""
time_length, episode_length, qtable,rewards = Q_learning(env, total_episodes, learning_rate, max_steps, gamma, epsilon,
                                                  max_epsilon, min_epsilon, decay_rate, verbose = True)
print("Number of episodes (size =30): ",episode_length)
print("Time (size = 30): ", time_length)
np.save("data/qtable30.npy",  qtable)
np.save("data/rewards30.npy",  rewards)
"""
qtable  = np.load('data/qtable30.npy')
rewards = np.load('data/rewards30.npy')

qbest = np.empty(env.nS)
#print(qtable)

for state in range(env.nS):

    qbest[state] = np.argmax(qtable[state,:])


print("Score over time: " + str(sum(rewards) / total_episodes))
#print(qtable)
fig, ax = plt.subplots()
rewards_moving = moving_average(rewards,100)
X = np.arange(1,len(rewards_moving)+1,1)

ax.plot(X, rewards_moving, color='steelblue', label="Mean Reward", linewidth=2.0, linestyle='-')
ax.legend(loc='best', frameon=True)
plt.grid(False, linestyle='--')
plt.title('Mean Reward vs number of episodes, Q learning (grid size = 30).')
plt.xlim((0,100000))
plt.ylabel('Mean Reward')
plt.xlabel(' Number of episodes')
plt.savefig('plots/lake_Qlearning_curve2.png')

for episode in range(5):
    state = env.reset()
    step = 0
    done = False
    #print("****************************************************")
    # print("EPISODE ", episode)

    for step in range(max_steps):

        # Take the action (index) that have the maximum expected future reward given that state
        action = np.argmax(qtable[state, :])
        new_state, reward, done, info = env.step(action)
        if done:
            # Here, we decide to only print the last state (to see if our agent is on the goal or fall into an hole)
            #env.render()

            # We print the number of step it took.
            #print("Number of steps", step)
            break
        state = new_state

#draw_policy(qtable, qbest, env, size, "plots/lake_Qvalue2.png",
#                "Policy visualization (Q learning), size = 30 ")
env.close
#
# Exploration Exploitation strategy
#
EPISODE_LENGTH =[]
TIME = []
DECAY_RATE = [ 0.0011,0.00105,0.0010,0.00095,0.0009,0.00085]
"""
for decay_rate in DECAY_RATE:
    np.random.seed(111)
    size = 30
    random_map = generate_random_map(size=size, p=0.85)
    env = gym.make("FrozenLake-v0", desc=random_map, is_slippery = False)
    RANDOM_SEED = 111

    env.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    random.seed(RANDOM_SEED)
    env.reset()
    env.render()
    env.action_space.seed(111)

    import random
    action_size = env.action_space.n
    state_size = env.observation_space.n
    qtable = np.zeros((state_size, action_size))

    total_episodes = 100000
    learning_rate = 0.7         # Learning rate
    max_steps = 100000           # Max steps per episode
    gamma = 0.6          # Discounting rate

    # Exploration parameters

    epsilon = 0.6                 # Exploration rate
    max_epsilon = 1.0             # Exploration probability at start
    min_epsilon = 10e-9           # Minimum exploration probability
    decay_rate = decay_rate       # Exponential decay rate for exploration prob

    # List of rewards
    rewards = []

    def moving_average(a, n=3) :
        ret = np.cumsum(a, dtype=float)
        ret[n:] = ret[n:] - ret[:-n]
        return ret[n - 1:] / n
    time_length, episode_length, qtable,rewards = Q_learning(env, total_episodes, learning_rate, max_steps, gamma, epsilon,
                                                  max_epsilon, min_epsilon, decay_rate, verbose = True)
    EPISODE_LENGTH.append(episode_length)
    TIME.append(time_length)
    qbest = np.empty(env.nS)
    #print(qtable)
    for state in range(env.nS):
        qbest[state] = np.argmax(qtable[state,:])
    print("Score over time: " + str(sum(rewards) / total_episodes))
    #print(qtable)
    env.close

np.save("data/time_eps.npy",  TIME)
np.save("data/episodes_eps.npy",  EPISODE_LENGTH)
"""
#EPISODE_LENGTH = np.array(EPISODE_LENGTH)
EPISODE_LENGTH = np.load('data/episodes_eps.npy')
TIME  = np.load('data/time_eps.npy')


# Create some mock data
x_labels = [ 0.0009, 0.00095,0.001]
x = np.arange(0,3,1)
y = [EPISODE_LENGTH[4], EPISODE_LENGTH[3], EPISODE_LENGTH[2] ]
z = [ TIME[4]         , TIME[3]          , TIME[2]           ]
label1 = ['Number of \n episodes']
label2 = ['Execution time']
fig, ax1 = plt.subplots()
plt.title("Execution time & Number of episodes for convergence \n(different decay rates).")
ax1.set_xlabel('Epsilon decay ')
ax1.set_ylabel('Number of episodes to converge')
ax1.bar(x-0.2, y, width=0.15, color='steelblue', align='center')
ax1.legend()
ax1.legend(label1, loc=1)

ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
ax2.set_ylabel('Execution Time (seconds)')  # we already handled the x-label with ax1
ax2.bar(x, z, width=0.15, color='orange', align='center')
ax2.legend()
ax2.legend(label2, loc=9)
fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.xticks(x, x_labels)

plt.savefig('plots/lake_diff_decayrates.png')
