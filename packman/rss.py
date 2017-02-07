import numpy as np
import random
import time
import pickle


def select_majority(a):
	c = {}
	for x in a:
		if x in c:
			c[x] += 1
		else:
			c[x] = 1
	mc = 0
	mx = c.keys()[0]
	for x in c:
		if mc < c[x]:
			mc, mx = c[x], x
	return mx


def load(f):
	r = pickle.load(open(f, 'rb'))
	print(len(r.triggers))
	return r


class RSSTrigger:
	def __init__(self, w, bias, action):
		self.w = w
		self.bias = bias
		self.action = action

	def is_triggered(self, obs):
		return self.w.ravel().dot(obs.ravel()) < self.bias

	def get_action(self):
		return self.action


class RSS:
	def __init__(self, step_fn, first_obs_fn, obs_dims, action_space=[0], cmr_repeats=9, alpha=0.9, alpha_decay=0.95):
		self.step_fn = step_fn
		self.first_obs_fn = first_obs_fn
		self.obs_dims = obs_dims
		self.action_space = action_space
		self.triggers = []
		self.cmr_repeats = cmr_repeats
		self.alpha = alpha
		self.alpha_decay = alpha_decay

	def get_random_action(self):
		return self.action_space[random.randrange(0, len(self.action_space))]

	def get_action(self, i):
		return self.action_space[i % len(self.action_space)]

	def react(self, obs):
		viable_actions = []
		for tr in self.triggers:
			if tr.is_triggered(obs):
				viable_actions.append(tr.get_action())
		return select_majority(viable_actions) if len(viable_actions) > 0 else self.get_random_action()

	def play_episode(self):
		r = 0
		obs = self.first_obs_fn()
		while True:
			action = self.react(obs)
			obs, reward, done = self.step_fn(action)
			r += reward
			if done:
				break
		return r

	def get_current_mean_reward(self, repeats):
		return np.float64([self.play_episode() for i in range(repeats)]).mean()

	def get_optimal_bias(self, w):
		obs = self.first_obs_fn()
		biases = []
		while True:
			biases.append(w.ravel().dot(obs.ravel()))
			action = self.react(obs)
			obs, _, done = self.step_fn(action)
			if done:
				break
		biases.sort()
		return biases[int(len(biases) * self.alpha)]

	def train(self, backup_intervals=60*2, trig_search_repeats=100, model_file_name='model.pkl'):
		last_backup_time = time.time()
		while True:
			cmr = self.get_current_mean_reward(self.cmr_repeats)
			print('Current mean reward: %f; learned triggers: %d; alpha: %.3f' % (cmr, len(self.triggers), self.alpha))
			if time.time() > last_backup_time + backup_intervals:
				last_backup_time = time.time()
				print('Saving model file...')
				pickle.dump(self, open(model_file_name, 'w'))
				print('Model saved.')
			best_reward = cmr
			best_trig = None
			assert(len(self.action_space) <= trig_search_repeats)
			for i in range(trig_search_repeats):
				w = np.random.normal(size=self.obs_dims)
				trig = RSSTrigger(w, self.get_optimal_bias(w), self.get_action(i))
				self.triggers.append(trig)
				reward = self.get_current_mean_reward(1)
				if best_reward < reward:
					reward = self.get_current_mean_reward(self.cmr_repeats)
					if best_reward < reward:
						best_reward = reward
						best_trig = trig
						print('New best reward: %f' % best_reward)
				self.triggers.pop()
			if best_trig is not None:
				self.triggers.append(best_trig)
			self.alpha *= self.alpha_decay
