import rss
import gym

env = gym.make('MsPacman-v0')

def first_obs_fn():
	return env.reset()

def step_fn(action):
	observation, reward, done, _ = env.step(action)
	return observation, reward, done

r = rss.load('model.pkl')

obs = first_obs_fn()

while True:
	action = r.react(obs)
	env.render()
	obs, reward, done = step_fn(action)
	if done:
		obs = env.reset()
