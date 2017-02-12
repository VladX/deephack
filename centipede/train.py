import rss
import gym
import argparse

env = gym.make('Centipede-v0')

def first_obs_fn():
	return env.reset()

def step_fn(action):
	observation, reward, done, _ = env.step(action)
	return observation, reward, done

parser = argparse.ArgumentParser()
parser.add_argument('--alpha', type=float, default=0.5)
parser.add_argument('--ad', type=float, default=0.96)
parser.add_argument('--bm', type=float, default=1.05)
parser.add_argument('--save', type=str, default='model.pkl')

args = parser.parse_args()

r = rss.RSS(step_fn, first_obs_fn, (250,160,3), range(18), 20, alpha=args.alpha, alpha_decay=args.ad, beta_multiplier=args.bm)
r.train(2 * 60, 500, args.save)
