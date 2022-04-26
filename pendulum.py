#Eva Domschot and Skyler Hughes
#sets up pendulum environment and prints observation and reward for random action

import numpy as np
import gym
import time 
from gym import wrappers
from keras.models import Model
from keras.layers import Dense, Flatten, Input, concatenate
from rl.agents import DDPGAgent
from rl.memory import SequentialMemory

    
def main():
	env = gym.make("Pendulum-v1")
	env.close()
    

	# Number of steps you run the agent for 
	num_steps = 100

	obs = env.reset()

	for step in range(num_steps):
		# take random action
		action = env.action_space.sample()

		# apply the action
		obs, reward, done, info = env.step(action)
		print('Observation: ', obs)
		print('Reward: ', reward)

		# Render the env
		env.render()

		# Wait a bit before the next frame unless you want to see a crazy fast video
		time.sleep(0.001)

		# If the epsiode is up, then start another one
		if done:
			env.reset()

	# Close the env
	env.close()
    
    
if __name__ == "__main__":
    main()
