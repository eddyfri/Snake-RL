import numpy as np
import os
import tensorflow as tf
from tqdm import trange

def training(iterations, env, agent, save_weights=True, save_path="weights/", file_name="weights.h5"):
    rewards_hist = []
    walls_hist = []
    fruit_hist = []
    loss_hist = []
    max_fruits_eaten = 0

    for iteration in trange(iterations):
        state = tf.constant(env.to_state())
        actions = agent.select_action(state)
        rewards = env.move(actions)
        new_state = tf.constant(env.to_state())
        dones = np.isin(rewards, [env.WIN_REWARD, env.HIT_WALL_REWARD, env.ATE_HIMSELF_REWARD])
        loss = agent.train(state, actions, rewards, new_state, dones)
        mean_reward = tf.reduce_mean(rewards).numpy()
        rewards_hist.append(mean_reward)
        if loss is not None:
            loss_hist.append(loss)
        wall_hits_count = np.sum(rewards == env.HIT_WALL_REWARD)
        walls_hist.append(wall_hits_count)

        fruits_eaten_count = np.sum(rewards == env.FRUIT_REWARD)
        fruit_hist.append(fruits_eaten_count)

        if iteration % 100 == 0:
            avg_reward = np.mean(rewards_hist[-100:]) if rewards_hist else 0
            avg_loss = np.mean(loss_hist[-100:]) if loss_hist else 0

            fruits_eaten = tf.reduce_sum(tf.cast(rewards == 0.5, tf.int32)).numpy()
            max_fruits_eaten = max(max_fruits_eaten, fruits_eaten)
            print(f"Iteration {iteration}: Avg Reward: {avg_reward:.2f}, Avg Loss: {avg_loss:.4f}")
            print(f"Fruits eaten: {fruits_eaten}, Max fruits eaten: {max_fruits_eaten}")
            print(f"Wall hits: {wall_hits_count}")

    if save_weights:
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        agent.save_weights(os.path.join(save_path, file_name))
        print(f"Weights saved to {os.path.join(save_path, file_name)}")

    return rewards_hist, walls_hist, fruit_hist, loss_hist