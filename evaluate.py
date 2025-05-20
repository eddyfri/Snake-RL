
import environments_fully_observable 
import environments_partially_observable
import numpy as np
import matplotlib.pyplot as plt
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import random

from tqdm import trange
from ddqn import DoubleDQNAgent
from baseline import heuristic_policy

tf.random.set_seed(0)
random.seed(0)
np.random.seed(0)

# function to standardize getting an env for the whole notebook
def get_env(n=1000, partially_observable=False):
    # n is the number of boards that you want to simulate parallely
    # size is the size of each board, also considering the borders
    # mask for the partially observable, is the size of the local neighborhood
    size = 7
    if partially_observable:
        e = environments_partially_observable.OriginalSnakeEnvironment(n, size, 3)
    else:
        e = environments_fully_observable.OriginalSnakeEnvironment(n, size)

    return e

def heuristic_evaluate(env, iterations=1000):
    heuristic_rewards = []
    fruits_eaten_heuristic = []
    wall_hits_heuristic = []

    for _ in trange(iterations):
        actions = heuristic_policy(env)
        rewards = env.move(actions)
        wall_hits_count = np.sum(rewards == env.HIT_WALL_REWARD)
        wall_hits_heuristic.append(wall_hits_count)
        fruits_eaten_count = np.sum(rewards == env.FRUIT_REWARD)
        fruits_eaten_heuristic.append(fruits_eaten_count)
        heuristic_rewards.append(np.mean(rewards))
    
    return heuristic_rewards, wall_hits_heuristic, fruits_eaten_heuristic

def evaluate(agent, env, iterations=1000):
    rewards_history = []
    wall_hits = []
    fruits_eaten = []

    for _ in trange(iterations):
        state = tf.constant(env.to_state())
        actions = agent.select_action(state)
        rewards = env.move(actions)
        wall_hits_count = np.sum(rewards == env.HIT_WALL_REWARD)
        wall_hits.append(wall_hits_count)
        fruits_eaten_count = np.sum(rewards == env.FRUIT_REWARD)
        fruits_eaten.append(fruits_eaten_count)
        rewards_history.append(np.mean(rewards))
        dones = np.isin(rewards.numpy().flatten(), [env.WIN_REWARD, env.HIT_WALL_REWARD, env.ATE_HIMSELF_REWARD])

    return rewards_history, wall_hits, fruits_eaten

def plots(rewards, fruits_eaten, wall_hits):
    rewards_history_ddqn, heuristic_rewards, random_rewards = rewards
    fruits_eaten_loaded_ddqn, fruits_eaten_heuristic = fruits_eaten
    wall_hits_ddqn, wall_hits_heuristic = wall_hits

    # PLOTS
    # Plot average rewards
    plt.figure(figsize=(10, 6))
    plt.bar(['DDQN', 'Heuristic', 'Random'], [np.mean(rewards_history_ddqn), np.mean(heuristic_rewards), np.mean(random_rewards)], color=['blue', 'orange', 'green'])
    plt.ylabel('Average Reward')
    plt.title('Average Rewards of DDQN, Heuristic, and Random')
    plt.grid(axis='y')
    plt.show()

    # Plot average wall hits
    plt.figure(figsize=(10, 6))
    plt.bar(['DDQN', 'Heuristic'], [np.mean(wall_hits_ddqn), np.mean(wall_hits_heuristic)], color=['blue', 'orange'])
    plt.ylabel('Average Wall Hits')
    plt.title('Average Wall Hits of DDQN and Heuristic')
    plt.grid(axis='y')
    plt.show()

    # Plot average fruits eaten
    plt.figure(figsize=(10, 6))
    plt.bar(['DDQN', 'Heuristic'], [np.mean(fruits_eaten_loaded_ddqn), np.mean(fruits_eaten_heuristic)], color=['blue', 'orange'])
    plt.ylabel('Average Fruits Eaten')
    plt.title('Average Fruits Eaten of DDQN and Heuristic')
    plt.grid(axis='y')
    plt.show()

if __name__ == "__main__":
    env_ = get_env()
    GAMMA = .9
    ITERATIONS = 100

    print("\n--- Evaluating DDQN ---")
    env_ = get_env()
    agent_ddqn = DoubleDQNAgent(state_shape=env_.to_state().shape[1:], gamma=GAMMA, n_actions=4, epsilon_start=1., epsilon_end=0.05, epsilon_decay=0.9995, learning_rate=1e-4, target_update_freq=100)
    agent_ddqn.load_weights("weights/snake_ddqn_weights.h5")
    agent_ddqn.epsilon = agent_ddqn.epsilon_end
    rewards_history_ddqn, wall_hits_ddqn, fruits_eaten_loaded_ddqn = evaluate(agent_ddqn, env_, iterations=ITERATIONS)

    print(f"Avg Reward - DDQN: {np.mean(rewards_history_ddqn):.2f}")
    print(f"Avg Wall Hits - DDQN: {np.mean(wall_hits_ddqn):.2f}")
    print(f"Avg Fruits Eaten - DDQN: {np.mean(fruits_eaten_loaded_ddqn):.2f}")

    # Heuristic evaluation
    print("\n--- Evaluating Heuristic ---")
    env_ = get_env()
    heuristic_rewards, wall_hits_heuristic, fruits_eaten_heuristic = heuristic_evaluate(env_, iterations=ITERATIONS)

    print(f"Avg Reward - Heuristic: {np.mean(heuristic_rewards):.2f}")
    print(f"Avg Wall Hits - Heuristic: {np.mean(wall_hits_heuristic):.2f}")
    print(f"Avg Fruits Eaten - Heuristic: {np.mean(fruits_eaten_heuristic):.2f}")

    # Random evaluation
    print("\n--- Evaluating Random ---")
    random_env = get_env()
    random_rewards = []

    for _ in trange(ITERATIONS):
        probs = tf.convert_to_tensor([[.25]*4]*random_env.n_boards)
        #sample actions
        actions =  tf.random.categorical(tf.math.log(probs), 1, dtype=tf.int32)
        # MDP update
        rewards = random_env.move(actions)
        random_rewards.append(np.mean(rewards))

    print(f"Avg Reward - Random: {np.mean(random_rewards):.2f}")

    rewards = [rewards_history_ddqn, heuristic_rewards, random_rewards]
    fruits_eaten = [fruits_eaten_loaded_ddqn, fruits_eaten_heuristic]
    wall_hits = [wall_hits_ddqn, wall_hits_heuristic]

    plots(rewards, fruits_eaten, wall_hits)
