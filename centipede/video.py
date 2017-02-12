import rss
import gym
from gym import wrappers
import sys
import random

random.seed(int(sys.argv[1]))

env = gym.make('Centipede-v0')
env = wrappers.Monitor(env, '/tmp/experiment-1')

def first_obs_fn():
	return env.reset()

def step_fn(action):
	observation, reward, done, _ = env.step(action)
	return observation, reward, done

r = rss.load('model.pkl')
f = open('rewards.log', 'w')

for i_episode in range(1):
	obs = first_obs_fn()
	while True:
		action = r.react(obs)
		#env.render()
		obs, reward, done = step_fn(action)
		f.write('%d\n' % (reward))
		if done:
			#print(i_episode)
			break
