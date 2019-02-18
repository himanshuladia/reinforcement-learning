import gym
import random
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

env_name = "CartPole-v1"

num_episodes = 10000
max_steps_per_episode = 100

learning_rate = 0.001
discount_rate = 0.95

exploration_rate = 1
max_exploration_rate = 1.
min_exploration_rate = 0.01
exploration_decay_rate = 0.995

BATCH_SIZE = 16
MEMORY_SIZE = 1000000


class DQNSolver:

	def __init__(self, state_space_size, action_space_size):
		self.exploration_rate = exploration_rate

		self.action_space_size = action_space_size
		self.state_space_size = state_space_size

		self.replay_memory = deque(maxlen=MEMORY_SIZE)

		# Model architecture
		self.model = Sequential()
		self.model.add(Dense(24, input_shape=(self.state_space_size,), activation='relu'))
		self.model.add(Dense(24, activation='relu'))
		self.model.add(Dense(self.action_space_size, activation='linear'))
		self.model.compile(loss='mse', optimizer=Adam(lr=learning_rate))

	def remember(self, state, action, reward, next_state, done):
		self.replay_memory.append((state, action, reward, next_state, done))

	def act(self, state):
		exploration_rate_threshold = random.uniform(0,1)
		if exploration_rate_threshold > exploration_rate:
			q_values = self.model.predict(state)
			action = np.argmax(q_values[0])
		else:
			action = random.randrange(self.action_space_size)
		return action

	def experience_replay(self):
		if len(self.replay_memory) < BATCH_SIZE:
			return
		batch = random.sample(self.replay_memory, BATCH_SIZE)
		for state, action, reward, next_state, done in batch:
			q_optimal = reward
			if done == False:
				q_optimal = reward + discount_rate * np.max(self.model.predict(next_state)[0])
			q_values = self.model.predict(state)
			q_values[0][action] = q_optimal
			self.model.fit(state, q_values, verbose=0)

		# Exploration rate decay
		self.exploration_rate *= exploration_decay_rate
		self.exploration_rate = max(min_exploration_rate, self.exploration_rate)


def cartpole():
	env = gym.make(env_name)

	state_space_size = env.observation_space.shape[0]
	action_space_size = env.action_space.n

	dqn_solver = DQNSolver(state_space_size, action_space_size)

	rewards_all_episodes = []

	for episode in range(num_episodes):
		state = env.reset()
		state = np.reshape(state, [1, state_space_size])

		step = 0

		while True:
			step += 1
			env.render()

			# Epsilon Greedy Strategy
			action = dqn_solver.act(state)

			# Take the action
			next_state, reward, done, info = env.step(action)
			next_state = np.reshape(next_state, [1, state_space_size])
			reward = reward if not done else -reward

			# Save in replay memory
			dqn_solver.remember(state, action, reward, next_state, done)

			# Update state
			state = next_state

			if done == True:
				print("Episode :" + str(episode+1) + ", Score :" + str(step))
				break

			# Perform experience replay
			dqn_solver.experience_replay()

if __name__=="__main__":
	cartpole()