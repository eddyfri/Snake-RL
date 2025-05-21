import environments_fully_observable
import environments_partially_observable
import matplotlib.pyplot as plt
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

np.random.seed(0)

from dqn import DQNAgent
from training import training

def get_env(n=1000, partially_observable=False):
    # n is the number of boards that you want to simulate parallely
    # size is the size of each board, also considering the borders
    # mask for the partially observable, is the size of the local neighborhood
    size = 7
    if partially_observable:
        e = environments_partially_observable.OriginalSnakeEnvironment(n, size, 3)
    else:
        e = environments_fully_observable.OriginalSnakeEnvironment(n, size)
    # or environments_partially_observable.OriginalSnakeEnvironment(n, size, 2)
    return e

if __name__ == "__main__":
    ITERATIONS = 5000
    env = get_env()
    # or env = get_env(partially_observable=True) if you want to test the partially observable environment
    agent = DQNAgent(state_shape=env.to_state().shape[1:], gamma=0.9, n_actions=4, epsilon_start=1., epsilon_end=0.05, epsilon_decay=0.9995, learning_rate=1e-4, target_update_freq=100)
    print("Starting DQN training...")
    rewards_history_dqn, wall_hits_dqn, fruits_eaten_dqn, _ = training(ITERATIONS, env, agent, save_weights=True, save_path="weights/", file_name="snake_dqn_weights.h5")

    print("DQN training completed.")

    # Results plotting
    plt.figure(figsize=(12, 6))
    plt.plot(rewards_history_dqn, label='DQN Rewards', color='blue')
    plt.title('DQN Training Rewards')
    plt.xlabel('Iterations')
    plt.ylabel('Average Reward')
    plt.legend()
    plt.grid()
    plt.show()

    plt.figure(figsize=(12, 6))
    plt.plot(wall_hits_dqn, label='DQN Wall Hits', color='red')
    plt.title('DQN Training Wall Hits')
    plt.xlabel('Iterations')
    plt.ylabel('Wall Hits')
    plt.legend()
    plt.grid()
    plt.show()
    
    plt.figure(figsize=(12, 6))
    plt.plot(fruits_eaten_dqn, label='DQN Fruits Eaten', color='green')
    plt.title('DQN Training Fruits Eaten')
    plt.xlabel('Iterations')
    plt.ylabel('Fruits Eaten')
    plt.legend()
    plt.grid()
    plt.show()
    