import numpy as np
import random
import os

# FAB data generation without the existence of POD
# The
num_lots = 138
num_scenarios = 200

for i in range(num_scenarios):
    data = 1800. * np.random.rand(num_lots)
    cmd_t = np.sort(data)

    missions = [(2, 3), (3, 2), (3, 6), (6, 3), (2, 6), (6, 2)]
    p23 = 1.5 / 19.
    p36 = 3. / 19.
    p62 = 5. / 19.
    prob = [p23, p23, p36, p36, p62, p62]

    from_to = random.choices(missions, weights=prob, k=num_lots)

    data_from, data_to = list(zip(*from_to))
    data_from = list(data_from)
    data_to = list(data_to)

    dir_path = './assets/scenario{}/'.format(i)
    os.mkdir(dir_path)

    np.save(dir_path + 'data_cmd.npy', np.array(cmd_t))
    np.save(dir_path + 'data_from.npy', np.array(data_from, dtype=np.int))
    np.save(dir_path + 'data_to.npy', np.array(data_to, dtype=np.int))

    data = np.array([cmd_t, data_from, data_to]).T
    np.savetxt(dir_path + 'data.csv', data, delimiter=',')
