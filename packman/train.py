import rss
import gym

env = gym.make('MsPacman-v0')

def first_obs_fn():
	return env.reset()

def step_fn(action):
	observation, reward, done, _ = env.step(action)
	return observation, reward, done

r = rss.RSS(step_fn, first_obs_fn, (210,160,3), range(9), alpha=0.2, alpha_decay=0.997)
r.train()
