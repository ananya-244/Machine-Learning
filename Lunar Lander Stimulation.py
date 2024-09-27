import time
from collections import deque, namedtuple

import gym
import numpy as np
import PIL.Image
import tensorflow as tf
import utils

from pyvirtualdisplay import Display
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.losses import MSE
from tensorflow.keras.optimizers import Adam

# Set up a virtual display to render the Lunar Lander environment.
Display(visible=0, size=(840, 480)).start()

# Set the random seed for TensorFlow
tf.random.set_seed(utils.SEED)

MEMORY_SIZE = 100_000     # size of memory buffer
GAMMA = 0.995             # discount factor
ALPHA = 1e-3              # learning rate
NUM_STEPS_FOR_UPDATE = 4  # perform a learning update every C time steps

env = gym.make('LunarLander-v2')

env.reset()
PIL.Image.fromarray(env.render(mode='rgb_array'))

state_size = env.observation_space.shape
num_actions = env.action_space.n

print('State Shape:', state_size)
print('Number of actions:', num_actions)

# Reset the environment and get the initial state.
current_state = env.reset()

# Select an action
action = 0

# Run a single time step of the environment's dynamics with the given action.
next_state, reward, done, _ = env.step(action)

# Display table with values.
utils.display_table(current_state, action, next_state, reward, done)

# Replace the `current_state` with the state after the action is taken
current_state = next_state



# Create the Q-Network
q_network = Sequential([
    Input(shape=state_size),
    Dense(units=64, activation='relu'),
    Dense(units=64, activation='relu'),
    Dense(units=num_actions, activation='linear'),
])

# Create the target Q^-Network
target_q_network = Sequential([
    Input(shape=state_size),
    Dense(units=64, activation='relu'),
    Dense(units=64, activation='relu'),
    Dense(units=num_actions, activation='linear'),
])

# Compile both networks
q_network.compile(optimizer=Adam(learning_rate=ALPHA), loss=MSE)
target_q_network.compile(optimizer=Adam(learning_rate=ALPHA), loss=MSE)

# Define experience tuples for replay buffer
experience = namedtuple("Experience", ["state", "action", "reward", "next_state", "done"])

# Initialize replay buffer
memory_buffer = deque(maxlen=MEMORY_SIZE)

# Function to learn from experiences
def agent_learn(experiences, gamma):
    # Unpack the experiences
    states, actions, rewards, next_states, dones = experiences

    # Compute the target Q values
    target_q_values = target_q_network.predict_on_batch(next_states)
    max_q_values = np.amax(target_q_values, axis=1)
    targets = rewards + gamma * (1 - dones) * max_q_values

    # Train the network with the updated Q-values
    masks = tf.one_hot(actions, num_actions)
    with tf.GradientTape() as tape:
        q_values = q_network(states)
        q_actions = tf.reduce_sum(tf.multiply(q_values, masks), axis=1)
        loss = MSE(targets, q_actions)

    grads = tape.gradient(loss, q_network.trainable_variables)
    q_network.optimizer.apply_gradients(zip(grads, q_network.trainable_variables))

# Training loop
num_episodes = 2000
total_point_history = []
epsilon = 1.0
num_p_av = 100

start = time.time()

for i in range(num_episodes):
    state = env.reset().reshape(1, -1)
    total_points = 0

    for t in range(1000):
        action = utils.get_action(q_network, state, epsilon, num_actions)
        next_state, reward, done, _ = env.step(action)
        next_state = next_state.reshape(1, -1)

        # Store experience tuple (S,A,R,S') in the memory buffer.
        memory_buffer.append(experience(state, action, reward, next_state, done))

        # Only update the network every NUM_STEPS_FOR_UPDATE time steps.
        update = utils.check_update_conditions(t, NUM_STEPS_FOR_UPDATE, memory_buffer)

        if update:
            # Sample random mini-batch of experience tuples (S,A,R,S') from D
            experiences = utils.get_experiences(memory_buffer)

            # Set the y targets, perform a gradient descent step,
            # and update the network weights.
            agent_learn(experiences, GAMMA)

        state = next_state.copy()
        total_points += reward

        if done:
            break

    total_point_history.append(total_points)
    av_latest_points = np.mean(total_point_history[-num_p_av:])

    # Update the ε value
    epsilon = utils.get_new_eps(epsilon)

    print(f"\rEpisode {i+1} | Total point average of the last {num_p_av} episodes: {av_latest_points:.2f}", end="")

    if (i+1) % num_p_av == 0:
        print(f"\rEpisode {i+1} | Total point average of the last {num_p_av} episodes: {av_latest_points:.2f}")

    # We will consider that the environment is solved if we get an
    # average of 200 points in the last 100 episodes.
    if av_latest_points >= 200.0:
        print(f"\n\nEnvironment solved in {i+1} episodes!")
        q_network.save('lunar_lander_model.h5')
        break

tot_time = time.time() - start

print(f"\nTotal Runtime: {tot_time:.2f} s ({(tot_time/60):.2f} min)")

# Plot the total point history along with the moving average
utils.plot_history(total_point_history)

# Suppress warnings from imageio
import logging
logging.getLogger().setLevel(logging.ERROR)

filename = "./videos/lunar_lander.mp4"

utils.create_video(filename, env, q_network)
utils.embed_mp4(filename)
