import rss
import gym

env = gym.make('Skiing-v0')

def first_obs_fn():
	return env.reset()

def step_fn(action):
	observation, reward, done, _ = env.step(action)
	return observation, reward, done

r = rss.RSS(step_fn, first_obs_fn, (250,160,3), [0,1,2], alpha=0.1, alpha_decay=0.995)
r.train(5,50)
