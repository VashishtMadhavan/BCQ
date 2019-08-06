import numpy as np
from sum_tree import SumSegmentTree, MinSegmentTree, LinearSchedule

# Code based on: 
# https://github.com/openai/baselines/blob/master/baselines/deepq/replay_buffer.py

# Simple replay buffer
class ReplayBuffer(object):
	def __init__(self, max_size=int(1e6)):
		self.storage = []
		self.max_size = max_size
		self.curr_idx = 0

	# Expects tuples of (state, next_state, action, reward, done)
	def add(self, data):
		if self.curr_idx < self.max_size:
			self.storage.append(data)
		else:
			idx = self.curr_idx % self.max_size
			self.storage[idx] = data
		self.curr_idx += 1

	def sample(self, batch_size):
		ind = np.random.randint(0, len(self.storage), size=batch_size)
		state, next_state, action, reward, done = [], [], [], [], []

		for i in ind: 
			s, s2, a, r, d = self.storage[i]
			state.append(np.array(s, copy=False))
			next_state.append(np.array(s2, copy=False))
			action.append(np.array(a, copy=False))
			reward.append(np.array(r, copy=False))
			done.append(np.array(d, copy=False))

		return (np.array(state), 
			np.array(next_state), 
			np.array(action), 
			np.array(reward).reshape(-1, 1), 
			np.array(done).reshape(-1, 1))

	def save(self, filename):
		np.save("./buffers/"+filename+".npy", self.storage)

	def load(self, filename):
		self.storage = np.load("./buffers/"+filename+".npy")

# Replay buffer with priority sampling
class PriorityReplayBuffer(object):
	def __init__(self, timesteps=int(1e6), alpha=0.6, beta=0.5, eps=1e-6, max_size=1e6):
		self.alpha = alpha
		self.beta_schedule = LinearSchedule(tsteps=timesteps, init_p=beta, final_p=1.0)
		self.eps = eps 
		self.storage = []
		self.max_size = max_size
		self.curr_idx = 0

		it_capacity = 1
		while it_capacity < int(1e6):
			it_capacity *= 2

		self._it_sum = SumSegmentTree(it_capacity)
		self._it_min = MinSegmentTree(it_capacity)
		self._max_priority = 1.0

	# Expects tuples of (state, next_state, action, reward, done)
	def add(self, data):
		if self.curr_idx < self.max_size:
			self.storage.append(data)
			idx = len(self.storage) - 1
		else:
			idx = self.curr_idx % self.max_size
			self.storage[idx] = data
		self.curr_idx += 1
		self._it_sum[idx] = self._max_priority ** self.alpha
		self._it_min[idx] = self._max_priority ** self.alpha

	def _sample_proportional(self, batch_size):
		ind = []
		while len(ind) < batch_size:
			mass = np.random.random() * self._it_sum.sum(0, len(self.storage) - 1)
			idx = self._it_sum.find_prefixsum_idx(mass)
			if idx not in ind:
				ind.append(idx)
		return ind

	def update_priorities(self, inds, priorities):
		assert len(inds) == len(priorities)
		for idx, p in zip(inds, priorities):
			self._it_sum[idx] = (p + self.eps) ** self.alpha
			self._it_min[idx] = (p + self.eps) ** self.alpha
			self._max_priority = max(self._max_priority, p)

	def sample(self, batch_size, t):
		ind = self._sample_proportional(batch_size)
		state, next_state, action, reward, done = [], [], [], [], []

		for i in ind: 
			s, s2, a, r, d = self.storage[i]
			state.append(np.array(s, copy=False))
			next_state.append(np.array(s2, copy=False))
			action.append(np.array(a, copy=False))
			reward.append(np.array(r, copy=False))
			done.append(np.array(d, copy=False))

		# update weights
		weights = []
		p_min = self._it_min.min() / self._it_sum.sum()
		beta = self.beta_schedule.value(t)
		max_weight = (p_min * len(self.storage)) ** (-beta)
		for idx in ind:
			p_sample = self._it_sum[idx] / self._it_sum.sum()
			weight = (p_sample * len(self.storage)) ** (-beta)
			weights.append(weight / max_weight)

		return (np.array(state), 
			np.array(next_state), 
			np.array(action), 
			np.array(reward).reshape(-1, 1), 
			np.array(done).reshape(-1, 1),
			np.array(weights).reshape(-1, 1),
			np.array(ind))

	def save(self, filename):
		np.save("./buffers/"+filename+".npy", self.storage)

	def load(self, filename):
		self.storage = np.load("./buffers/"+filename+".npy")