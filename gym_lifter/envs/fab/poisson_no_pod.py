import numpy as np
import random
import os
from tqdm import tqdm


# FAB data generation without the existence of POD


num_lots = 6624                 # expected number of arrived lots during a single simulation
sim_time = 86400.               # data for 24 hrs

# parameter \lambda for Poisson process
# S_n : time when n-th lot arrival occurs
# Here we assume S_n = X_1 + ... X_n, where each X_k is an i.i.d. sample drawn from Exponential(\lambda):
# P(X_k > x) = e^{-\lambda x}, x > 0
# N_t : # of lots arrived until time t which is defined by
# N_t = \max(n >= 0 : S_n <=> t)
# Since N_t >= n iff S_n <= t, we deduce that N_t follows Poisson(\lambda t)
# Thus, E(N_t) = \lambda t
# See also Billingsley, 1979.
lam = num_lots / sim_time
beta = 1. / lam     # for numpy exponential distribution
missions = [(2, 3), (3, 2), (3, 6), (6, 3), (2, 6), (6, 2)]
p23 = 1.5 / 19.
p36 = 3. / 19.
p62 = 5. / 19.
prob = [p23, p23, p36, p36, p62, p62]


num_scenarios = 200     # number of total scenarios


for i in tqdm(range(num_scenarios)):
    cmd_t = []
    num_arrival = 0
    elapsed_t = 0.
    while elapsed_t < sim_time:
        dt = np.random.exponential(beta)
        elapsed_t += dt
        cmd_t.append(elapsed_t)
        num_arrival += 1

    cmd_t.pop()
    num_arrival -= 1
    from_to = random.choices(missions, weights=prob, k=num_arrival)

    data_from, data_to = list(zip(*from_to))
    data_from = list(data_from)
    data_to = list(data_to)

    dir_path = './assets/day/scenario{}/'.format(i)
    os.mkdir(dir_path)

    np.save(dir_path + 'data_cmd.npy'.format(i), np.array(cmd_t))
    np.save(dir_path + 'data_from.npy'.format(i), np.array(data_from, dtype=np.int))
    np.save(dir_path + 'data_to.npy'.format(i), np.array(data_to, dtype=np.int))

    # data = np.array([cmd_t, data_from, data_to]).T
    # np.savetxt(dir_path + 'data.csv'.format(i), data, delimiter=',')
