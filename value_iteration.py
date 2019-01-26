import numpy as np
import gym
import random
import time
import os

env = gym.make("FrozenLake-v0")
action_space_size = env.action_space.n
state_space_size = env.observation_space.n

q_table = np.zeros((state_space_size, action_space_size))

num_episodes = 10000
max_steps_per_episode = 100

learning_rate = 0.1
discount_rate = 0.99

exploration_rate = 1
max_exploration_rate = 1
min_exploration_rate = 0.01
exploration_decay_rate = 0.001

rewards_all_episodes = []

# Q Learning algorithm
for episode in range(num_episodes):
	
	state = env.reset()
	done = False
	rewards_current_episode = 0

	for step in range(max_steps_per_episode):

		# Epsilon Greedy Strategy
		exploration_rate_threshold = random.uniform(0,1)
		if exploration_rate_threshold > exploration_rate:
			action = np.argmax(q_table[state,:])
		else:
			action = env.action_space.sample()
		new_state, reward, done, info = env.step(action)

		# Update Q-value
		q_table[state,action] = (1-learning_rate)*q_table[state, action] + learning_rate*(reward+discount_rate*np.max(q_table[new_state,:]))

		# Update state and rewards
		state = new_state
		rewards_current_episode += reward

		if done == True:
			break

	# Exploration rate decay
	exploration_rate = min_exploration_rate + (max_exploration_rate-min_exploration_rate)*np.exp(-exploration_decay_rate*episode)

	rewards_all_episodes.append(rewards_current_episode)

	# Logging the rewards
	if (episode+1)%1000 == 0:
		print("Average reward after", episode+1, "episodes :", sum(rewards_all_episodes[-1000:])/1000)

# Print the final Q-table
print(q_table)


# Play the game
for episode in range(3):

	state = env.reset()
	done = False
	print("Episode number ", episode+1)
	time.sleep(1)

	for step in range(max_steps_per_episode):
		os.system('clear')
		env.render()
		time.sleep(0.3)

		action = np.argmax(q_table[state,:])
		new_state, reward, done, info = env.step(action)

		if done == True:
			os.system('clear')
			env.render()
			if reward == 1:
				print("You reached the goal!")
				time.sleep(3)
			else:
				print("You fell into the hoal!")
				time.sleep(3)
			os.system('clear')
			break

		state = new_state

env.close()