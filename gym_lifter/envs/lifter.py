import gym
import numpy as np
from os import path
from gym_lifter.envs.conveyor import Wafer, ConveyorBelt
from gym_lifter.envs.rack import Rack
from typing import Optional, Dict, Tuple


class LifterEnv(gym.Env):
	def __init__(self):
		# super(gym.Env, self).__init__()
		self._NUM_FLOORS = 3
		self._NUM_LAYERS = 3 * 3
		self._NUM_CONVEYORS = 3 * 3 * 3

		self.data_cmd = np.load(path.join(path.dirname(__file__), "cmd_time.npy"))
		self.data_from = np.load(path.join(path.dirname(__file__), "departure.npy"))
		self.data_to = np.load(path.join(path.dirname(__file__), "destination.npy"))

		self.num_data = self.data_cmd.shape[0]		# number of total wafers arrived during the episode
		self.newly_added = None

		self._BEGIN = None
		self._END = None

		# information relevant to rack master
		self.content: Optional[Wafer] = None		# object of class Wafer will be assigned or None if it is empty
		self.rack_position = None

		self.rack = Rack()

		self.floors = None

		self.t = None
		self.dt = 0.15

		self.conveyors: Dict[int, ConveyorBelt] = {}		# family of InConveyors labelled by their floors
		# self.conveyors: Dict[Tuple[int, int, int], ConveyorBelt] = {}		# TODO : to be extended to multi-layer case

		self._STATE_DIM = 4 + 2 * self._NUM_FLOORS
		self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self._STATE_DIM,), dtype=np.float)
		########################
		# action 0 : DOWN(-1)  #
		# action 1 : UP(+1)  #
		# action 2 : STAY, LOAD LOWER    #
		# action 3 : STAY, LOAD UPPER
		# action 4 : STAY, LOAD BOTH
		########################
		# TODO : add actions
		self.action_space = gym.spaces.Discrete(5)
		self.state = None

		return

	def step(self, action):

		assert self.action_space.contains(action)
		if action > 1:
			# if the lifter stops, load / release the wafer, or just stay
			# logic for release
			# always release first, and then load or do something else
			if self.rack.is_lower_loaded:
				if self.rack.destination[0] == self.rack_position:
					# release the wafer
					self.rack.release_lower_fork()
				else:
					pass
			elif self.rack.is_upper_loaded:
				if self.rack.destination[1] == self.rack_position + 1:
					# release the wafer
					self.rack.release_upper_fork()
				else:
					pass

			# logic for load
			if action == 2:
				# lower
				if self.conveyors[self.rack_position].is_empty:
					pass
				else:
					# load a wafer from the queue
					self.rack.load_lower(self.conveyors[self.rack_position].pop())
			elif action == 3:
				# upper
				if self.rack_position + 1 < self._NUM_FLOORS:
					if self.conveyors[self.rack_position + 1].is_empty:
						pass
					else:
						# load a wafer from the queue
						self.rack.load_upper(self.conveyors[self.rack_position + 1].pop())
			elif action == 4:
				# both
				if self.conveyors[self.rack_position].is_empty:
					pass
				else:
					# load a wafer from the queue
					self.rack.load_lower(self.conveyors[self.rack_position].pop())
				if self.rack_position + 1 < self._NUM_FLOORS:
					if self.conveyors[self.rack_position + 1].is_empty:
						pass
					else:
						# load a wafer from the queue
						self.rack.load_upper(self.conveyors[self.rack_position + 1].pop())

		else:
			self.rack_position = max(min(self._NUM_FLOORS, self.rack_position + (2 * action - 1)), 1)

		wt = self.waiting_time
		reward = -np.sum(wt**2)
		d1, d2 = self.rack.destination
		self.state = np.zeros(self._STATE_DIM)
		self.state[0:4] = float(self.rack.is_lower_loaded), float(d1), float(d2), float(self.rack_position)
		self.state[4:4 + self._NUM_FLOORS] = wt
		self.state[4 + self._NUM_FLOORS:] = self.destination
		self.simulate_arrival()		# Queue update during (t, t + dt]

		return self.state, reward, False, {}

	def reset(self):
		self._BEGIN = 0
		self._END = 0
		self.t = 0.
		self.rack_position = np.random.randint(low=1, high=self._NUM_FLOORS + 1)
		self.content = None
		# maps each floor number to the corresponding conveyor belt
		self.conveyors = {floor: ConveyorBelt() for floor in range(1, self._NUM_FLOORS + 1)}
		self.state = np.zeros(self._STATE_DIM)
		self.state[3] = self.rack_position

		return self.state

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
		for i in range(self._NUM_FLOORS, 0, -1):
			print('Floor{} | '.format(i), end='')
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
			print('| {:>5.1f}s [{:<3}]'.format(60 * (self.waiting_time[i - 1]), len(self.conveyors[i])), end='')
			print('|{}F'.format(self.destination[i - 1]), end='')
			print('|'.format(i), len(self.conveyors[i]) * '*')
		print('\n')
		return


if __name__ == '__main__':
	# test
	env = LifterEnv()
	env.reset()
	for _ in range(100):
		a = np.random.randint(low=0, high=5)
		print('action selected = ', a)
		_, _, _, _ = env.step(a)
		env.render()
