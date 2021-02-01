import numpy as np
import random


num_lots = 138

data = 1800. * np.random.rand(num_lots)
cmd_t = np.sort(data)

missions = [(2, 3), (3, 2), (3, 6), (6, 3), (2, 6), (6, 2)]
prob = [0.1, 0.1, 0.25, 0.25, 0.15, 0.15]

from_to = random.choices(missions, weights=prob, k=num_lots)


data_from, data_to = list(zip(*from_to))
data_from = list(data_from)
data_to = list(data_to)

np.save('./assets/data_cmd.npy', np.array(cmd_t))
np.save('./assets/data_from.npy', np.array(data_from, dtype=np.int))
np.save('./assets/data_to.npy', np.array(data_to, dtype=np.int))

data = np.array([cmd_t, data_from, data_to]).T
np.savetxt('./assets/data.csv', data, delimiter=',')
