import numpy as np
import os
import random
import tensorflow as tf
from collections import deque

class DQN(tf.keras.Model):
    def __init__(self, input_dim, n_actions):
        super(DQN, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='valid')
        self.conv2 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='valid')
        self.flatten = tf.keras.layers.Flatten()
        self.dense = tf.keras.layers.Dense(256, activation='relu')
        self.out = tf.keras.layers.Dense(n_actions, activation=None)

    def call(self, x):
        x = tf.cast(x, tf.float32)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.flatten(x)
        x = self.dense(x)
        return self.out(x) 

class ReplayBuffer:
    def __init__(self, size=10000):
        self.buffer = deque(maxlen=size)        

    def add(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        if len(self.buffer) < batch_size:
            batch_size = len(self.buffer)
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        return (
            np.array(states),
            np.array(actions),
            np.array(rewards),
            np.array(next_states),
            np.array(dones)
        )
    
    def __len__(self):
        return len(self.buffer)

    
class DoubleDQNAgent:
    def __init__(self, state_shape, n_actions=4, buffer_size=10000, batch_size=128, epsilon_start=1., epsilon_end=0.1, epsilon_decay=0.9995, learning_rate=1e-4, gamma=0.9, target_update_freq=100):
        self.state_shape = state_shape
        self.n_actions = n_actions
        self.batch_size = batch_size
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.lr = learning_rate
        self.target_update_freq = target_update_freq
        self.optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=self.lr)
        self.step_counter = 0

        self.qnet = DQN(state_shape, n_actions)
        self.target_network = DQN(state_shape, n_actions)

        # Initialization with a forward pass
        dummy_state = np.zeros((1,) + state_shape)
        self.qnet(dummy_state)
        self.target_network(dummy_state)

        self.update_target_network()
        self.replay_buffer = ReplayBuffer(buffer_size)

    def update_target_network(self):
        """ Copy weights from main Q-network to target network """
        self.target_network.set_weights(self.qnet.get_weights())

    def select_action(self, state):
        """ Select action using epsilon-greedy policy """
        batch_size = state.shape[0]
        if np.random.random() < self.epsilon:
            actions = np.random.randint(0, self.n_actions, size=(batch_size, 1))
        else:
            q_values = self.qnet(state)
            actions = tf.argmax(q_values, axis=1, output_type=tf.int32)
            actions = tf.reshape(actions, [-1, 1])
            actions = actions.numpy()

        return actions
    
    def store_experience(self, state, action, reward, next_state, done):
        """ Store experience in replay buffer """
        for i in range(state.shape[0]):
            self.replay_buffer.add((
                state[i].numpy(),
                action[i][0],
                reward[i].numpy()[0],
                next_state[i].numpy(),
                done[i]
            ))
    
    @tf.function
    def _compiled_train_step(self, states, actions, rewards, next_states, dones):
        with tf.GradientTape() as tape:
            q_values = self.qnet(states)
            action_mask = tf.one_hot(actions, self.n_actions)
            q_values_for_actions = tf.reduce_sum(q_values * action_mask, axis=1)

            # 1. Use main network to select best action for next state
            next_q_values_main = self.qnet(next_states)
            next_actions = tf.argmax(next_q_values_main, axis=1, output_type=tf.int32)
            # 2. Use target network to evaluate those actions
            next_q_values_target = self.target_network(next_states)
            max_next_q = tf.gather(next_q_values_target, next_actions, batch_dims=1)
            dones = tf.cast(dones, tf.float32)

            target_q_values = rewards + (1 - dones) * self.gamma * max_next_q

            # Loss            
            loss = tf.keras.losses.MSE(target_q_values, q_values_for_actions)

        gradients = tape.gradient(loss, self.qnet.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.qnet.trainable_variables))

        return loss
    
    def train(self, state, action, reward, next_state, done):
        """ Store experience and train the Q-network """
        self.store_experience(state, action, reward, next_state, done)

        if len(self.replay_buffer) < self.batch_size:
            return 0.0
        
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)

        states = tf.convert_to_tensor(states, dtype=tf.float32)
        actions = tf.convert_to_tensor(actions, dtype=tf.int32)
        rewards = tf.convert_to_tensor(rewards, dtype=tf.float32)
        next_state = tf.convert_to_tensor(next_states, dtype=tf.float32)
        dones = tf.convert_to_tensor(dones, dtype=tf.bool)

        loss = self._compiled_train_step(states, actions, rewards, next_state, dones)

        self.step_counter += 1
        if self.step_counter % self.target_update_freq == 0:
            self.update_target_network()
        
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

        return loss.numpy()
    
    def get_q_values(self, state):
        """ Get Q-values for a given state """
        return self.qnet(state)
    
    def save_weights(self, filepath="snake_dqn_weights.h5"):
        """ Save the model weights """
        self.qnet.save_weights(filepath)
        print(f"Model weights saved to {filepath}")

    def load_weights(self, filepath="snake_dqn_weights.h5"):
        if not os.path.exists(filepath):
            raise ValueError(f"File {filepath} does not exist.")
        self.qnet.load_weights(filepath)
        self.epsilon = self.epsilon_end
        self.update_target_network()
        print(f"Model weights loaded from {filepath}")