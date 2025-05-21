import environments_fully_observable
import environments_partially_observable
import matplotlib.pyplot as plt
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from ddqn import DoubleDQNAgent
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
    agent = DoubleDQNAgent(state_shape=env.to_state().shape[1:], n_actions=4, buffer_size=10000, batch_size=128, epsilon_start=1., epsilon_end=0.1, epsilon_decay=0.9995, learning_rate=1e-4, gamma=0.9, target_update_freq=100)
    print("Starting Double DQN training...")
    rewards_history_ddqn, wall_hits_ddqn, fruits_eaten_ddqn, _ = training(ITERATIONS, env, agent, save_weights=True, save_path="weights/", file_name="snake_ddqn_weights.h5")

    print("Double DQN training completed.")

    # Results plotting
    plt.figure(figsize=(12, 6))
    plt.plot(rewards_history_ddqn, label='DDQN Rewards', color='blue')
    plt.title('DDQN Training Rewards')
    plt.xlabel('Iterations')
    plt.ylabel('Average Reward')
    plt.legend()
    plt.grid()
    plt.show()

    plt.figure(figsize=(12, 6))
    plt.plot(wall_hits_ddqn, label='DDQN Wall Hits', color='red')
    plt.title('DDQN Training Wall Hits')
    plt.xlabel('Iterations')
    plt.ylabel('Wall Hits')
    plt.legend()
    plt.grid()
    plt.show()

    plt.figure(figsize=(12, 6))
    plt.plot(fruits_eaten_ddqn, label='DDQN Fruits Eaten', color='green')
    plt.title('DDQN Training Fruits Eaten')
    plt.xlabel('Iterations')
    plt.ylabel('Fruits Eaten')
    plt.legend()
    plt.grid()
    plt.show()