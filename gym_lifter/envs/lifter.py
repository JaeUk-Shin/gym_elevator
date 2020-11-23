import gym
import numpy as np
from os import path
from gym_lifter.envs.conveyor import Wafer, ConveyorBelt
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

		# information relevant to rack master
		self.content: Optional[Wafer] = None		# object of class Wafer will be assigned
		self.lifter_position = None
		self.is_full: bool = False

		self.floors = None

		self.t = None
		self.dt = 0.15

		self._BEGIN = None
		self._END = None

		self.conveyors: Dict[int, ConveyorBelt] = {}		# family of InConveyors labelled by their floors
		# self.conveyors: Dict[Tuple[int, int, int], ConveyorBelt] = {}		# TODO : to be extended to multi-layer case

		self._STATE_DIM = 3 + self._NUM_FLOORS
		self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self._STATE_DIM,), dtype=np.float)

		########################
		# action 0 : DOWN(-1)  #
		# action 1 : STAY(0)   #
		# action 2 : UP(+1)    #
		########################
		self.action_space = gym.spaces.Discrete(3)

		self.state = None

		return

	def step(self, action):

		assert self.action_space.contains(action)
		if action == 1:		# if the lifter stops, load / release the wafer, or just stay
			if self.is_full:
				if self.content.destination == self.lifter_position:
					# release the wafer
					self.content = None
					self.is_full = False
				else:
					pass
			else:
				if self.conveyors[self.lifter_position].is_empty:
					pass
				else:
					# load a wafer from the queue
					self.content = self.conveyors[self.lifter_position].pop()
					self.is_full = True

		self.lifter_position = max(min(self._NUM_FLOORS, self.lifter_position + (action - 1)), 1)
		wt = self.waiting_time
		reward = -np.sum(wt**2)
		dest = None
		if self.content is None:
			dest = 0			# TODO : need to be fixed
		else:
			dest = self.content.destination
		self.state = np.zeros(self._STATE_DIM)
		self.state[0], self.state[1], self.state[2] = float(self.is_full), float(dest), float(self.lifter_position)
		self.state[3:] = wt
		self.simulate_arrival()		# Queue update during (t, t + dt]

		return self.state, reward, False, {}

	def reset(self):
		self._BEGIN = 0
		self._END = 0
		self.t = 0.
		self.is_full = False
		self.lifter_position = np.random.randint(low=1, high=self._NUM_FLOORS + 1)
		self.content = None
		self.conveyors = {}			# maps each floor number to the corresponding conveyor belt
		for floor in range(1, self._NUM_FLOORS + 1):
			self.conveyors[floor] = ConveyorBelt()
		self.state = np.zeros(3 + self._NUM_FLOORS,)
		self.state[2] = self.lifter_position

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

	def render(self, mode='human'):
		print('elapsed time = {:.0f} sec'.format(60. * self.t))
		print('{} wafers have arrived ({} newly added)'.format(self._BEGIN, self.newly_added))
		for i in range(self._NUM_FLOORS, 0, -1):
			print('Floor{} | '.format(i), end='')
			# denote the position of the elevator & whether elevator is filled
			if self.lifter_position == i:
				print('@', end='')
				if self.is_full:
					print('o ', end='')
				else:
					print('  ', end='')
			else:
				print('   ', end='')
			print('| {:.1f}s [{:<3}]'.format(60 * (self.waiting_time[i - 1]), len(self.conveyors[i])), end='')
			print('|'.format(i), len(self.conveyors[i]) * '*')
		print('\n')
		return


if __name__ == '__main__':
	# test
	env = LifterEnv()
	env.reset()
	for _ in range(100):
		a = np.random.randint(low=0, high=3)
		_, _, _, _ = env.step(a)
		env.render()
