import numpy as np
import random
import os

# FAB data generation without the existence of POD
# The
num_lots = 6624
num_scenarios = 200

for i in range(num_scenarios):
    data = 86400. * np.random.rand(num_lots)
    cmd_t = np.sort(data)

    missions = [(2, 3, 0), (3, 2, 0), (3, 6, 0), (6, 3, 0), (2, 6, 0), (6, 2, 0), (2, 6, 1), (6, 2, 1)]
    p23 = 1.5 / 20.
    p36 = 3. / 20.
    p62 = 5. / 20.
    pod = .5 / 20.
    prob = [p23, p23, p36, p36, p62, p62, pod, pod]

    from_to_pod = random.choices(missions, weights=prob, k=num_lots)

    data_from, data_to, data_pod = list(zip(*from_to_pod))
    data_from = list(data_from)
    data_to = list(data_to)
    data_pod = list(data_pod)

    dir_path = '../assets/day_pod_uniform/scenario{}/'.format(i)
    os.mkdir(dir_path)

    np.save(dir_path + 'data_cmd.npy', np.array(cmd_t))
    np.save(dir_path + 'data_from.npy', np.array(data_from, dtype=np.int))
    np.save(dir_path + 'data_to.npy', np.array(data_to, dtype=np.int))
    np.save(dir_path + 'data_pod.npy', np.array(data_pod, dtype=np.int))

    """
    data = np.array([cmd_t, data_from, data_to]).T
    np.savetxt(dir_path + 'data.csv', data, delimiter=',')
    """
