import environments_fully_observable
import environments_partially_observable
import numpy as np

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
    rewards_history_ddqn, wall_hits_ddqn, fruits_eaten_ddqn, _ = training(ITERATIONS, env, agent, save_weights=True, save_path="weights/", file_name="snake_ddqn_weights.h5")

    print("Double DQN training completed.")

    # Results
    print("Double DQN training results:")
    print(f"Average rewards: {np.mean(rewards_history_ddqn)}")
    print(f"Average wall hits: {np.mean(wall_hits_ddqn)}")
    print(f"Average fruits eaten: {np.mean(fruits_eaten_ddqn)}")