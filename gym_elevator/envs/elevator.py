import gym
import numpy as np
from os import path


class ElevatorEnv(gym.Env):
	def __init__(self):
		# super(gym.Env, self).__init__()

		self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(6,), dtype=np.float)

		################################
		# action 0 : DOWN              #
		# action 1 : STAY              #
		# action 2 : UP                #
		################################
		self.action_space = gym.spaces.Discrete(3)

		self.state = None


		self.cmd_time = np.load(path.join(path.dirname(__file__), "cmd_time.npy"))
		self.departure = np.load(path.join(path.dirname(__file__), "departure.npy"))
		self.destination = np.load(path.join(path.dirname(__file__), "destination.npy"))

		self.content = None

		self.elevator_pos = None
		self.is_full = None

		self.t = None
		self.dt = 0.15

		self.begin = 0

		self.floors = None

		return

	def step(self, action):

		prev_t = self.t - self.dt
		end = self.begin

		while prev_t <= self.cmd_time[end] < self.t:
			end += 1

		for idx in range(self.begin, end):
			cmd_t = self.cmd_time[idx]
			depart = self.departure[idx]
			to = self.destination[idx]
			self.floors[depart].append((cmd_t, to))

		if action == 1:
			if self.is_full:
				if self.content[1] == self.elevator_pos:
					self.content = None		# release elevator
				else:
					pass
			else:
				if len(self.floors[self.elevator_pos]) == 0:
					pass
				else:
					self.content = self.floors[self.elevator_pos].pop(0)

		self.elevator_pos = max(min(3, self.elevator_pos + (action - 1)), 1)

		if len(self.floors[1]) == 0:
			waiting1 = 0.
		else:
			waiting1 = self.t - self.floors[1][0][0]

		if len(self.floors[2]) == 0:
			waiting2 = 0.
		else:
			waiting2 = self.t - self.floors[2][0][0]
		if len(self.floors[3]) == 0:
			waiting3 = 0.
		else:
			waiting3 = self.t - self.floors[3][0][0]

		reward = -(waiting1**2 + waiting2**2 + waiting3**2)
		done = False

		dest = None
		if self.content == None:
			dest = 0
		else:
			dest = self.content[1]

		self.state = np.array([float(self.is_full), dest, self.elevator_pos, waiting1, waiting2, waiting3])

		self.t += self.dt
		self.begin = end

		return self.state, reward, done, {}

	def reset(self):
		self.t = self.dt
		self.is_full = False
		self.elevator_pos = 2		# elevator begins to operate at the 2nd floor

		self.content = None

		self.floors = {1: [], 2: [], 3: []}
		self.state = np.array([0., 0, self.elevator_pos, 0., 0., 0.])
		return self.state

	def render(self, mode='human'):
		return

if __name__ == '__main__':
	"""
	arrival = np.genfromtxt('arrival.csv', delimiter=',', dtype=None)

	cmd_time = np.array([t[0] for t in arrival])
	departure = np.array([t[1] for t in arrival])
	destination = np.array([t[2] for t in arrival])

	np.save('cmd_time.npy', cmd_time)
	np.save('departure.npy', departure)
	np.save('destination.npy', destination)
	"""

	env = ElevatorEnv()
