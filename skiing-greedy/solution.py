import gym
from gym import wrappers
import numpy as np
from skimage.measure import label, regionprops, compare_mse
import random

SKIER_COLOR = np.int64((214, 92, 92))
INDICATOR_COLOR = np.int64((66, 72, 200))
INDICATOR_COLOR_FINISH = np.int64((184, 50, 50))

SOME_BOTVA = np.int64((256 * 256, 256, 1))

EPS = 10
MAGIC_CONST_1 = 10

env = gym.make('Skiing-v0')
#env = wrappers.Monitor(env, '/tmp/experiment-1')

def rgb_to_int64(img):
	img = np.int64(img)
	return img.dot(SOME_BOTVA)

def int64_to_rgb(n):
	return np.int64((int(n / 256**2) % 256, int(n / 256) % 256, int(n) % 256))

def VZHUUH_i_paramy(ind):
	if len(ind) % 2 == 1:
		min_dist = float('inf')
		p = None
		for p1 in ind:
			for p2 in ind:
				if p1 != p2:
					dist = (p1[0] - p2[0])**2 + (p1[1] - p2[1])**2
					if dist < min_dist:
						min_dist = dist
						p = (p1, p2)
		ind.remove(p[0])
		ind.remove(p[1])
		ind.append(((p[0][0] + p[1][0]) * 0.5, (p[0][1] + p[1][1]) * 0.5))
	return ind

def locate_objs(img):
	intensity = rgb_to_int64(img)
	labels = label(intensity)
	props = regionprops(labels, intensity)
	skier = []
	ind = []
	final_ind_coords = []
	for p in props:
		if compare_mse(int64_to_rgb(p.min_intensity), SKIER_COLOR) < EPS:
			skier += [p.centroid]
		if compare_mse(int64_to_rgb(p.min_intensity), INDICATOR_COLOR) < EPS or compare_mse(int64_to_rgb(p.min_intensity), INDICATOR_COLOR_FINISH) < EPS:
			ind += [p.centroid]
	skier = np.mean(skier, 0)

	if len(ind) % 2 == 1: # WTF?!
		ind = VZHUUH_i_paramy(ind)
		assert(len(ind) % 2 == 0)

	ind.sort()
	for i in range(0, len(ind), 2):
		p1, p2 = ind[i], ind[i+1]
		p1, p2 = np.float32(p1), np.float32(p2)
		final_ind_coords += [(p1 + p2) * 0.5]

	return skier, final_ind_coords

last_ac = -1
in_a_row = 0

def YOLO_skiing_strategy(observation, t):
	global last_ac
	global in_a_row
	skier, ind = locate_objs(observation)
	if len(ind) == 0:
		return 0
	for p in ind:
		if p[0] > skier[0]:
			break
	ac = 0
	if p[1] < skier[1]:# - 10:
		ac = 2
	elif p[1] > skier[1]:# + 10:
		ac = 1
	if t % 30 == 0:
		in_a_row = max(1, in_a_row - 1)
	if (last_ac == ac and in_a_row >= MAGIC_CONST_1) or ac == 0:
		return 0
	elif last_ac == ac:
		in_a_row += 1
		return ac
	else:
		last_ac = ac
		in_a_row = 1
		return ac

for i_episode in range(100):
	observation = env.reset()
	for t in range(10000):
		action = YOLO_skiing_strategy(observation, t)
		env.render()
		observation, reward, done, _ = env.step(action)
		print reward
		if done:
			print("Episode finished %d" % t)
			break
	#break
