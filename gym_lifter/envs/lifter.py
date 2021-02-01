import gym
import numpy as np
from gym_lifter.envs.action_set import available_actions, action2operation
from gym_lifter.envs.fab.fab import FAB


class LifterEnv(gym.Env):
	# TODO : modularize a FAB to keep the code simple & readable
	def __init__(self):
		# super(gym.Env, self).__init__()

		self.fab = FAB()
		self.state_dim = 4 + 2 * self.fab.num_layers
		# state variables : rack position, pod, lower_to, upper_to, con_i_to's, con_i_wt's
		# note that operating time is not regarded as a state variable
		# TODO : take POD into consideration
		self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.state_dim,), dtype=np.float)
		# 24 actions in total
		self.action_space = gym.spaces.Discrete(30)
		self.state = None
		self.pos_to_flr = [2, 2, 2, 3, 3, 3, 3, 6, 6, 6]  # convert control point to floor
		return

	def reset(self):
		self.fab.reset()
		return self.get_obs()

	def step(self, action: int):
		assert self.action_space.contains(action)
		if action == 29:
			operation = None
		else:
			operation = action2operation[action]

		wt = self.waiting_time
		rew = -np.sum(wt ** 2) / 3600.
		# operate the FAB
		info = self.fab.sim(operation)

		return self.get_obs(), rew, False, info

	def render(self, mode='human'):
		"""
		print('elapsed time = {:.0f} sec'.format(60. * self.t))
		print('{} wafers have arrived ({} newly added)'.format(self.BEGIN, self.newly_added))

		for i in range(self._NUM_FLOORS + 1, 0, -1):
			print('Floor{:<2} | '.format(i), end='')
			# denote the position of the elevator & whether elevator is filled
			if self.rack_position + 1 == i:
				print('@', end='')
				if self.rack.is_upper_loaded:
					print('* ', end='')
				else:
					print('  ', end='')
			elif self.rack_position == i:
				print('@', end='')
				if self.rack.is_lower_loaded:
					print('* ', end='')
				else:
					print('  ', end='')
			else:
				print('   ', end='')
			if i == self._NUM_FLOORS + 1:
				print('|')
			else:
				print('| {:>5.1f}s [{:<3}]'.format(60 * (self.waiting_time[i - 1]), len(self.conveyors[i])), end='')
				print('|{}F'.format(self.destination[i - 1]), end='')
				print('|'.format(i), len(self.conveyors[i]) * '*')
		print('\n')
		"""
		return

	def get_obs(self) -> np.ndarray:
		# encode the current state into a point in $\mathbb{R}^n$ (n : state space dimension)
		normalized_pos = self.rack_pos / 9.
		rack_flr = self.pos_to_flr[self.rack_pos]
		lower_to, upper_to = self.rack_destination
		rack_info = np.array(
			[normalized_pos, float(self.is_pod_loaded), (lower_to - rack_flr) / 4., (upper_to - rack_flr) / 4.])
		obs = np.concatenate([rack_info, self.travel_distance, self.normalized_wt])
		return obs

	@staticmethod
	def action_map(state):
		return available_actions(state)

	@property
	def waiting_time(self):
		return self.fab.waiting_time

	@property
	def normalized_wt(self):
		return self.waiting_time / 180.

	@property
	def rack_pos(self):
		return self.fab.rack_pos

	@property
	def destination(self):
		return self.fab.destination

	@property
	def rack_destination(self):
		return self.fab.rack_destination

	@property
	def is_pod_loaded(self):
		return self.fab.is_pod_loaded

	@property
	def travel_distance(self):
		return self.fab.travel_distance


if __name__ == '__main__':
	# test
	env = LifterEnv()
	env.reset()
	reward = 0.
	for _ in range(100):
		a = np.random.randint(low=0, high=5)
		print('action selected = ', a)
		_, r, _, _ = env.step(a)
		reward += r
		env.render()
	print('reward : {}'.format(reward))
