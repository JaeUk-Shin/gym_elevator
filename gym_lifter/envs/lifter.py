import gym
import numpy as np
from typing import List, Tuple
from gym_lifter.envs.action_set import action2operation, available_actions_no_wt
from gym_lifter.envs.fab.fab import FAB


class LifterEnv(gym.Env):
	def __init__(self):
		# super(gym.Env, self).__init__()

		self.fab = FAB(mode='day')
		self.num_layers = self.fab.num_layers
		self.state_dim = 3 + 2 * self.fab.num_layers
		# state variables : rack position, lower_to, upper_to, con_i_to's, con_i_wt's
		# note that operating time is not regarded as a state variable
		self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.state_dim,), dtype=np.float)
		# 24 actions in total
		self.action_space = gym.spaces.Discrete(26)
		self.state = None
		self.pos_to_flr = [2, 2, 2, 3, 3, 3, 3, 6, 6, 6]  # convert control point to floor
		self.capacities = self.fab.capacities
		self.action2operation = action2operation
		return

	def reset(self, mode=None):
		self.fab.reset(mode=mode)
		return self.get_obs()

	def step(self, action: int):
		assert self.action_space.contains(action)
		if action == 25:
			operation = None
		else:
			operation = action2operation[action]

		if action in [11, 12, 13, 14, 16, 17, 18, 19, 21, 22, 23]:
			rew = 1.
		elif action in [15, 20, 24]:
			rew = 2.
		else:
			rew = 0.

		# wt = self.waiting_time
		# rew = -np.sum(wt) / 3600.
		# operate the FAB
		info = self.fab.sim(operation)
		done = False
		return self.get_obs(), rew, done, info

	def get_obs(self) -> np.ndarray:
		# encode the current state into a point in $\mathbb{R}^n$ (n : state space dimension)
		######################################################################################################
		# rack position | POD | lower destination | upper destination | queue destination | waiting quantity #
		######################################################################################################
		rpos = self.rack_pos / 9.
		lower_to, upper_to = self.rack_destination
		# rack_info = [rpos, float(self.is_pod_loaded), lower_to / 6., upper_to / 6.]
		rack_info = [rpos, lower_to / 6., upper_to / 6.]
		destination = [d / 6. for d in self.destination]
		waiting_quantity = [(self.waiting_quantity[i] / self.capacities[i]) ** 5 for i in range(self.num_layers)]
		layer_info = destination + waiting_quantity
		obs = np.array(rack_info + layer_info)
		return obs

	@staticmethod
	def action_map(state) -> List[int]:
		return available_actions_no_wt(state)

	@property
	def operation_log(self):
		return self.fab.operation_log

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
	def destination(self) -> List[int]:
		return self.fab.destination

	@property
	def rack_destination(self) -> Tuple[int, int]:
		return self.fab.rack_destination

	@property
	def is_pod_loaded(self):
		return self.fab.is_pod_loaded

	@property
	def travel_distance(self):
		return self.fab.travel_distance

	@property
	def waiting_quantity(self) -> List[int]:
		return self.fab.waiting_quantity

	def render(self, mode='human'):
		self.fab.render()
		return

	def close(self):
		self.fab.close()
