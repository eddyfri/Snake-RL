import numpy as np
import tensorflow as tf

tf.random.set_seed(0)
np.random.seed(0)

class A2C(tf.keras.Model):
    def __init__(self, input_dim, n_actions):
        super(A2C, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='valid')
        self.conv2 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='valid')
        self.flatten = tf.keras.layers.Flatten()
        self.shared = tf.keras.layers.Dense(256, activation='relu')

        self.policy_logits = tf.keras.layers.Dense(n_actions) # actor head
        self.value = tf.keras.layers.Dense(1) # critic head

    def call(self, x):
        x = tf.cast(x, tf.float32)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.flatten(x)
        x = self.shared(x)
        return self.policy_logits(x), self.value(x)
    

class A2CAgent:
    def __init__(self, state_shape, n_actions, gamma=0.99, learning_rate=5e-5, entropy_beta=0.0001):
        self.n_actions = n_actions
        self.gamma = gamma
        self.entropy_beta = entropy_beta

        self.model = A2C(state_shape, n_actions)
        dummy = tf.zeros((1,) + state_shape)
        self.model(dummy) # init

        self.optimizer = tf.keras.optimizers.Adam(learning_rate)

    def select_action(self, state):
        logits, _ = self.model(state)
        action_dist = tf.random.categorical(logits, 1)
        return action_dist.numpy()
    
    def compute_returns(self, rewards, dones, last_value):
        returns = np.zeros_like(rewards)
        R = last_value
        for t in reversed(range(len(rewards))):
            R = rewards[t] + self.gamma * R * (1 - dones[t])
            returns[t] = R
        return returns
    
    def train(self, state, actions, rewards, next_state, dones):
        with tf.GradientTape() as tape:
            logits, values = self.model(state)
            _, next_value = self.model(next_state)

            values = tf.squeeze(values)
            next_value = tf.squeeze(next_value)

            returns = self.compute_returns(rewards.numpy().flatten(), dones, next_value[-1].numpy())

            advantages = returns - values.numpy()

            advantages = (advantages - np.mean(advantages)) / (np.std(advantages) + 1e-8) # normalize advantages, 1e-8 to avoid division by zero

            action_masks = tf.one_hot(actions.flatten(), self.n_actions)
            log_probs = tf.nn.log_softmax(logits)
            log_action_probs = tf.reduce_sum(log_probs * action_masks, axis=1)
            policy_loss = -tf.reduce_mean(log_action_probs * advantages)

            value_loss = tf.reduce_mean(tf.square(returns - values))
            entropy = -tf.reduce_sum(tf.nn.softmax(logits) * log_probs, axis=1)
            entropy_loss = tf.reduce_mean(entropy)

            total_loss = policy_loss + 0.5 * value_loss - self.entropy_beta * entropy_loss

        grads = tape.gradient(total_loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

        return total_loss.numpy()
    
    def save_weights(self, filepath):
        self.model.save_weights(filepath)
        print(f"Model weights saved to {filepath}")

    def load_weights(self, filepath):
        self.model.load_weights(filepath)
        print(f"Model weights loaded from {filepath}")

    