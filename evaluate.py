
import environments_fully_observable 
import environments_partially_observable
import numpy as np
import matplotlib.pyplot as plt
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import random

from tqdm import trange
from dqn import DQNAgent
from ddqn import DoubleDQNAgent
from a2c import A2CAgent


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

def heuristic_policy(env):
    UP, RIGHT, DOWN, LEFT = 0, 1, 2, 3
    ACTIONS = [UP, RIGHT, DOWN, LEFT]

    DIRS = {
        UP: (-1, 0),
        RIGHT: (0, 1),
        DOWN: (1, 0),
        LEFT: (0, -1)
    }

    boards = env.boards
    n_boards, board_size, _ = boards.shape
    actions = []

    for i in range(n_boards):
        board = boards[i]
        head = tuple(map(int, np.argwhere(board == env.HEAD)[0]))
        fruit = tuple(map(int, np.argwhere(board == env.FRUIT)[0]))
        candidates = []
        for action in ACTIONS:
            dy, dx = DIRS[action]
            next_pos = (head[0] + dy, head[1] + dx)

            if(0 <= next_pos[0] < board_size) and (0 <= next_pos[1] < board_size):
                target_cell = board[next_pos]
                if target_cell != env.WALL and target_cell != env.BODY:
                    distance = abs(next_pos[0] - fruit[0]) + abs(next_pos[1] - fruit[1])
                    candidates.append((action, distance))
        
        if not candidates:
            action = np.random.choice(ACTIONS)
        else:
            candidates.sort(key=lambda x: x[1])
            action = candidates[0][0]
        actions.append(action)
    
    return tf.convert_to_tensor(actions, dtype=tf.int32)[:, None]

def heuristic_evaluate(env, iterations=1000):
    heuristic_rewards = []
    fruits_eaten_heuristic = []
    wall_hits_heuristic = []

    for iteration in trange(ITERATIONS):
        actions = heuristic_policy(env)
        rewards = env.move(actions)
        wall_hits_count = np.sum(rewards == env.HIT_WALL_REWARD)
        wall_hits_heuristic.append(wall_hits_count)
        fruits_eaten_count = np.sum(rewards == env.FRUIT_REWARD)
        fruits_eaten_heuristic.append(fruits_eaten_count)
        heuristic_rewards.append(np.mean(rewards))
        dones = np.isin(rewards.numpy().flatten(), [env.WIN_REWARD, env.HIT_WALL_REWARD, env.ATE_HIMSELF_REWARD])

        if np.sum(dones) > actions.shape[0] / 2:
            env = get_env() # if all the boards are done, reset the environment
    
    return heuristic_rewards, wall_hits_heuristic, fruits_eaten_heuristic

def evaluate(agent, env, iterations=1000):
    rewards_history = []
    wall_hits = []
    fruits_eaten = []

    for iteration in trange(iterations):
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

if __name__ == "__main__":
    env_ = get_env()
    GAMMA = .9
    ITERATIONS = 5000

    agent_dqn = DQNAgent(state_shape=env_.to_state().shape[1:], gamma=GAMMA, n_actions=4, epsilon_start=1., epsilon_end=0.05, epsilon_decay=0.9995, learning_rate=1e-4, target_update_freq=100)

    agent_dqn.load_weights("weights/snake_dqn_weights.h5")
    agent_dqn.epsilon = agent_dqn.epsilon_end

    rewards_history_dqn, wall_hits_dqn, fruits_eaten_loaded_dqn = evaluate(agent_dqn, env_, iterations=ITERATIONS)

    print(f"Avg Reward - DQN: {np.mean(rewards_history_dqn):.2f}")
    print(f"Avg Wall Hits - DQN: {np.mean(wall_hits_dqn):.2f}")
    print(f"Avg Fruits Eaten - DQN: {np.mean(fruits_eaten_loaded_dqn):.2f}")

    env_ = get_env()
    agent_ddqn = DoubleDQNAgent(state_shape=env_.to_state().shape[1:], gamma=GAMMA, n_actions=4, epsilon_start=1., epsilon_end=0.05, epsilon_decay=0.9995, learning_rate=1e-4, target_update_freq=100)
    agent_ddqn.load_weights("weights/snake_ddqn_weights.h5")
    agent_ddqn.epsilon = agent_ddqn.epsilon_end
    rewards_history_ddqn, wall_hits_ddqn, fruits_eaten_loaded_ddqn = evaluate(agent_ddqn, env_, iterations=ITERATIONS)

    print(f"Avg Reward - DDQN: {np.mean(rewards_history_ddqn):.2f}")
    print(f"Avg Wall Hits - DDQN: {np.mean(wall_hits_ddqn):.2f}")
    print(f"Avg Fruits Eaten - DDQN: {np.mean(fruits_eaten_loaded_ddqn):.2f}")

    env_ = get_env()
    agent_a2c = A2CAgent(state_shape=env_.to_state().shape[1:], gamma=GAMMA, n_actions=4, learning_rate=1e-4, entropy_beta=0.01)
    agent_a2c.load_weights("weights/snake_a2c_weights.h5")

    rewards_history_a2c, wall_hits_a2c, fruits_eaten_loaded_a2c = evaluate(agent_a2c, env_, iterations=ITERATIONS)

    print(f"Avg Reward - A2C: {np.mean(rewards_history_a2c):.2f}")
    print(f"Avg Wall Hits - A2C: {np.mean(wall_hits_a2c):.2f}")
    print(f"Avg Fruits Eaten - A2C: {np.mean(fruits_eaten_loaded_a2c):.2f}")

    # Heuristic evaluation
    env_ = get_env()
    heuristic_rewards, wall_hits_heuristic, fruits_eaten_heuristic = heuristic_evaluate(env_, iterations=ITERATIONS)

    print(f"Avg Reward - Heuristic: {np.mean(heuristic_rewards):.2f}")
    print(f"Avg Wall Hits - Heuristic: {np.mean(wall_hits_heuristic):.2f}")
    print(f"Avg Fruits Eaten - Heuristic: {np.mean(fruits_eaten_heuristic):.2f}")

    # Random evaluation
    random_env = get_env()
    random_rewards = []

    for _ in trange(ITERATIONS):
        probs = tf.convert_to_tensor([[.25]*4]*random_env.n_boards)
        #sample actions
        actions =  tf.random.categorical(tf.math.log(probs), 1, dtype=tf.int32)
        # MDP update
        rewards = random_env.move(actions)
        random_rewards.append(np.mean(rewards))

    # Plot average rewards
    plt.figure(figsize=(10, 6))
    plt.bar(['DQN', 'DDQN', 'A2C', 'Heuristic', 'Random'], [np.mean(rewards_history_dqn), np.mean(rewards_history_ddqn), np.mean(rewards_history_a2c), np.mean(heuristic_rewards), np.mean(random_rewards)], color=['blue', 'orange', 'green', 'red', 'purple'])
    plt.ylabel('Average Reward')
    plt.title('Average Rewards of DQN, DDQN, A2C, Heuristic, and Random')
    plt.grid(axis='y')
    plt.savefig('plots/average_rewards_comparison.png')
    plt.show()

    # Plot average wall hits
    plt.figure(figsize=(10, 6))
    plt.bar(['DQN', 'DDQN', 'A2C', 'Heuristic'], [np.mean(wall_hits_dqn), np.mean(wall_hits_ddqn), np.mean(wall_hits_a2c), np.mean(wall_hits_heuristic)], color=['blue', 'orange', 'green', 'red'])
    plt.ylabel('Average Wall Hits')
    plt.title('Average Wall Hits of DQN, DDQN, A2C, and Heuristic')
    plt.grid(axis='y')
    plt.savefig('plots/average_wall_hits_comparison.png')
    plt.show()

    # Plot average fruits eaten
    plt.figure(figsize=(10, 6))
    plt.bar(['DQN', 'DDQN', 'A2C', 'Heuristic'], [np.mean(fruits_eaten_loaded_dqn), np.mean(fruits_eaten_loaded_ddqn), np.mean(fruits_eaten_loaded_a2c), np.mean(fruits_eaten_heuristic)], color=['blue', 'orange', 'green', 'red'])
    plt.ylabel('Average Fruits Eaten')
    plt.title('Average Fruits Eaten of DQN, DDQN, A2C, and Heuristic')
    plt.grid(axis='y')
    plt.savefig('plots/average_fruits_eaten_comparison.png')
    plt.show()

    # Plot images from training that are in plots folder
    images = os.listdir('plots')
    images = [img for img in images if img.endswith('.png') and 'average' not in img]
    images.sort()
    fig, axs = plt.subplots(1, len(images), figsize=(15, 5))
    for i, img in enumerate(images):
        img_path = os.path.join('plots', img)
        image = plt.imread(img_path)
        axs[i].imshow(image)
        axs[i].axis('off')
        axs[i].set_title(img.split('.')[0])
    plt.tight_layout()
    plt.savefig('plots/training_images.png')
    plt.show()
