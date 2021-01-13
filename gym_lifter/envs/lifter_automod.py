import gym
import numpy as np
import os
from os import path
from gym_lifter.envs.conveyor import Wafer, ConveyorBelt
from gym_lifter.envs.rack import Rack
from typing import Dict, List, Any
from gym_lifter.envs.launch_server import launch_server


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
		self.ctrl_pts = [1, 2, 3, 4, 5, 6, 8, 9, 10]

		# create socket to communicate with Automod programs
		self.socket = launch_server()

		return

	def step(self, a: int):
		assert self.action_space.contains(a)
		# action : 0 ~ 35 (36 actions in total)
		# each action is of the form (to what floor(1, 2, 3, 4, 5, 6, 8, 9, 10), load / unload, upper / lower)
		floor_id, remainder = a // 4, a % 4
		destination = self.ctrl_pts[floor_id]

		load, fork = remainder // 2, remainder % 2
		"""
		# send the action to execute in txt format
		if os.path.exists('actions.txt'):
			os.remove('actions.txt')
		message = open('action.txt', 'w')
		message.write('{} {} {}'.format(destination, load, fork))
		message.close()
		"""
		ctrl_signal = ('{} {} {}'.format(destination, load, fork)).encode()
		self.socket.send(ctrl_signal)
		"""
		while not os.path.isfile('obs.txt'):
			# TODO : I think this is not a good idea
			pass
		obs_file = open('obs.txt', 'r')
		"""
		obs_data: str = self.socket.recv(8192).decode()		# receive state information from the simulation
		obs_list: List[Any] = obs_data.split()
		dt = obs_list[-1]
		obs_list.append(destination)
		obs = np.array(obs_list)
		wt = obs[2:11]

		rew = -np.sum(wt**2)
		self.t += dt

		return obs, rew, False, {'dt': dt}

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

	def reset(self):
		# self._BEGIN = 0
		# self._END = 0
		self.t = 0.
		self.rack_position = np.random.randint(low=1, high=self.NUM_FLOORS)
		# maps each floor number to the corresponding conveyor belt

		self.state = np.zeros(self.STATE_DIM)
		self.state[self.STATE_DIM - 1] = self.rack_position
		# TODO : send a message to Automod program so that it can reset the simulation
		return self.state
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