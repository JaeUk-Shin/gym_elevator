import gym
import numpy as np
import os
from os import path
from gym_lifter.envs.conveyor import Wafer, ConveyorBelt
from gym_lifter.envs.rack import Rack
from typing import Dict, List, Any, Sequence, Union
from gym_lifter.envs.launch_server import launch_server
import sys
import time


class AutomodLifterEnv(gym.Env):
	def __init__(self):
		# super(gym.Env, self).__init__()
		self.NUM_FLOORS = 9

		"""
		self._NUM_LAYERS = 3 * 3
		self._NUM_CONVEYORS = 3 * 3 * 3

		self.data_cmd = np.load(path.join(path.dirname(__file__), "assets/floor{}/data_cmd.npy".format(self.NUM_FLOORS)))
		self.data_from = np.load(path.join(path.dirname(__file__), "assets/floor{}/data_from.npy".format(self.NUM_FLOORS)))
		self.data_to = np.load(path.join(path.dirname(__file__), "assets/floor{}/data_to.npy".format(self.NUM_FLOORS)))
		# self.data_is_pod = np.load(path.join(path.dirname(__file__), "assets/data_is_pod.npy"))

		self.num_data = self.data_cmd.shape[0]		# number of total wafers arrived during the episode
		self.newly_added = None

		# self._BEGIN = None
		# self._END = None
	
		# information relevant to rack master
		self.rack_position = None
		self.rack = Rack()

		self.floors = None
		"""
		self.t = None
		# self.dt = 0.15

		self.conveyors: Dict[int, ConveyorBelt] = {}		# family of InConveyors labelled by their floors
		# self.conveyors: Dict[Tuple[int, int, int], ConveyorBelt] = {}		# TODO : to be extended to multi-layer case

		self.STATE_DIM = 2 + 2 * self.NUM_FLOORS + 1 + 1
		################################################################################################################
		# STATE VARIABLES                                                                                              #
		# +----------------------------------------------------------------------------------------------------------+ #
		# | To(Lower) | To(Upper) | Waiting Time(each Conveyor) | To(each Conveyor) | Operating Time | Rack Position | #
		# +----------------------------------------------------------------------------------------------------------+ #
		################################################################################################################
		# TODO : take POD into consideration
		self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.STATE_DIM,), dtype=np.float)
		self.action_space = gym.spaces.Discrete(36)
		# self.state = None
		self.ctrl_pts = [1, 2, 3, 4, 5, 6, 7, 8, 9]

		# create socket to communicate with Automod programs
		self.socket = launch_server()
		self.t_record = None

		self.state_bytes = None
		self.prefix = None
		return

	def step(self, action: int):
		assert self.action_space.contains(action)
		# action : 0 ~ 35 (36 actions in total)
		# each action is of the form (to what floor(1, 2, 3, 4, 5, 6, 8, 9, 10), load / unload, upper / lower)
		floor_id, rem = action // 4, action % 4
		destination = self.ctrl_pts[floor_id]

		load, fork = rem // 2, rem % 2

		ctrl_signal: bytes = ('{} {} {} '.format(destination, load, fork)).encode()
		ctrl_to_send = self.prefix + ctrl_signal + self.state_bytes
		print('control sent : ', ctrl_signal)
		print('\n')
		self.socket.send(ctrl_to_send)

		data: bytes = self.socket.recv(8192)		# receive state information from the simulation
		self.prefix = data[:2]
		state_list = self.bytes2list(data)
		self.state_bytes = ' '.join(state_list[2:]).encode()
		t_updated = float(state_list[-1])
		dt = t_updated - self.t_record
		state_list.append(destination)
		state = np.array(state_list, dtype='float')
		print('state received : ', state)
		print('available actions : ', self.admissible_actions(state, mask=False))
		print('available actions (in number) : ', [[action // 4 + 1, (action % 4) // 2, (action % 4) % 2] for action in self.admissible_actions(state, mask=False)])
		wt = state[2:11]

		rew = -np.sum(wt**2)
		self.t = t_updated

		return state, rew, False, {'dt': dt}

	"""
	def step_tmp(self, action):
		assert self.action_space.contains(action)
		if action > 1:
			# if the lifter stops, load / release the wafer, or just stay
			# always release first, and then load or do something else
			if self.rack.destination[0] == self.rack_position:
				# release the wafer from the lower fork if exists
				self.rack.release_lower_fork()
			if self.rack.destination[1] == self.rack_position + 1:
				# release the wafer from the upper fork if exists
				self.rack.release_upper_fork()

			# logic for load
			if action == 2:
				self.load_lower()
			elif action == 3:
				self.load_upper()
			elif action == 4:
				self.load_lower(), self.load_upper()
		else:
			# lifter moves up/ down
			self.rack_position = max(min(self._NUM_FLOORS, self.rack_position + (2 * action - 1)), 1)

		wt = self.waiting_time
		reward = -np.sum(wt**2)
		d1, d2 = self.rack.destination
		self.state = np.zeros(self._STATE_DIM)
		self.state[0: 5] = float(self.rack.is_lower_loaded), float(self.rack.is_upper_loaded), float(d1), float(d2), float(self.rack_position)
		self.state[5: 5 + self._NUM_FLOORS] = wt
		self.state[5 + self._NUM_FLOORS:] = self.destination
		self.simulate_arrival()		# Queue update during (t, t + dt]

		return self.state, reward, False, {}
	"""

	@staticmethod
	def bytes2list(data: bytes) -> list:
		return data[2:].decode().split()[1:]

	def reset(self):
		# self._BEGIN = 0
		# self._END = 0
		self.t = 0.
		rack_position = np.random.randint(low=1, high=self.NUM_FLOORS)
		# maps each floor number to the corresponding conveyor belt

		state = np.zeros(self.STATE_DIM)
		state[self.STATE_DIM - 2] = rack_position
		# TODO : send a message to Automod program so that it can reset the simulation
		data = self.socket.recv(8192)

		self.prefix: bytes = data[:2]
		print('prefix : ', self.prefix)
		state_list = self.bytes2list(data)
		self.state_bytes = ' '.join(state_list[2:]).encode()
		self.t_record = float(state_list[-1])
		state_list.append(8)		# starts at floor 8

		state = np.array(state_list, dtype=float)
		print('state received : ', state)
		return state
	"""
	def simulate_arrival(self):
		# read data for simulation
		next_t = self.t + self.dt
		while self._END < self.num_data:
			if self.t < self.data_cmd[self._END] <= next_t:
				self._END += 1
			else:
				break
		self.newly_added = self._END - self._BEGIN
		for i in range(self._BEGIN, self._END):
			# wafer generation from data
			wafer = Wafer(cmd_t=self.data_cmd[i], origin=self.data_from[i], destination=self.data_to[i])
			self.conveyors[self.data_from[i]].push(wafer)		# add load to the queue
		self.t += self.dt
		self._BEGIN = self._END

		return

	def load_lower(self):
		if self.conveyors[self.rack_position].is_empty:
			return
		else:
			# load a wafer from the queue
			self.rack.load_lower(self.conveyors[self.rack_position].pop())
		return

	def load_upper(self):
		if self.rack_position + 1 <= self._NUM_FLOORS:
			if self.conveyors[self.rack_position + 1].is_empty:
				return
			else:
				# load a wafer from the queue
				self.rack.load_upper(self.conveyors[self.rack_position + 1].pop())
		return

	@property
	def waiting_time(self):
		wt = np.zeros(self._NUM_FLOORS)
		for floor in range(1, self._NUM_FLOORS + 1):
			if self.conveyors[floor].is_empty:
				wt[floor - 1] = 0.
			else:
				wt[floor - 1] = self.t - self.conveyors[floor].cmd_time
		return wt

	@property
	def destination(self):
		return np.array([self.conveyors[i].destination for i in range(1, self._NUM_FLOORS + 1)])

	def render(self, mode='human'):
		print('elapsed time = {:.0f} sec'.format(60. * self.t))
		print('{} wafers have arrived ({} newly added)'.format(self._BEGIN, self.newly_added))

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
		return
	"""

	@staticmethod
	def admissible_actions(state, mask: bool = True) -> Union[np.ndarray, Sequence[int]]:
		"""return a set of admissible actions at a given state"""
		# unpack state variables
		lower = int(round(state[0]))
		upper = int(round(state[1]))
		wt = state[2:11]					# waiting time of the foremost lot of each queue

		# action set construction
		action_set = []
		if lower == 0:						# go to 2F/3F/6F & load the lower fork
			action_set += [4 * k + 2 for k in range(9) if wt[k] > 0.]
		elif lower == 2:
			action_set += [0, 4, 8] 		# go to 2F & unload a lot from the lower fork
		elif lower == 5:
			action_set += [12, 16, 20]		# go to 3F & unload a lot from the lower fork
		elif lower == 9:
			action_set += [24, 28, 32]		# go to 6F & unload a lot from the lower fork

		if upper == 0:						# go to 2F/3F/6F & load the upper fork
			action_set += [4 * k + 3 for k in range(9) if wt[k] > 0.]
		elif upper == 2:
			action_set += [1, 5, 9]  		# go to 2F & unload a lot from the upper fork
		elif upper == 5:
			action_set += [13, 17, 21]  	# go to 3F & unload a lot from the upper fork
		elif upper == 9:
			action_set += [25, 29, 33]  	# go to 6F & unload a lot from the upper fork
		if not mask:
			return action_set
		else:
			mask = np.full(36, -np.inf)		# generate a mask representing the set
			mask[action_set] = 0.								# 0 if admissible,  -inf else

			return mask


if __name__ == '__main__':
	env = AutomodLifterEnv()
	s = env.reset()
	for i in range(100):
		print('Received : ', s)
		a = env.action_space.sample()
		s, r, d, info = env.step(a)
