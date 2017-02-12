import rss
import gym
from gym import wrappers

env = gym.make('Centipede-v0')
env = wrappers.Monitor(env, '/tmp/experiment-1')

def first_obs_fn():
	return env.reset()

def step_fn(action):
	observation, reward, done, _ = env.step(action)
	return observation, reward, done

r = rss.load('model.pkl')

for i_episode in range(100):
	obs = first_obs_fn()
	while True:
		action = r.react(obs)
		#env.render()
		obs, reward, done = step_fn(action)
		if done:
			print(i_episode)
			break
