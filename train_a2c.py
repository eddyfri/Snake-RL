import environments_fully_observable
import environments_partially_observable
import numpy as np

from a2c import A2CAgent
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
    agent = A2CAgent(state_shape=env.to_state().shape[1:], n_actions=4, gamma=0.9, learning_rate=5e-5, entropy_beta=0.0001)
    rewards_history_ac, wall_hits_ac, fruits_eaten_ac, _ = training(ITERATIONS, env, agent, save_weights=True, save_path="weights/", file_name="snake_a2c_weights.h5")

    print("A2C training completed.")

    # Results
    print("A2C training results:")
    print(f"Average rewards: {np.mean(rewards_history_ac)}")
    print(f"Average wall hits: {np.mean(wall_hits_ac)}")
    print(f"Average fruits eaten: {np.mean(fruits_eaten_ac)}")