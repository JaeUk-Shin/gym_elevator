from gym_lifter.envs.fab.wafer import Wafer
from gym_lifter.envs.fab.rack import Rack
from gym_lifter.envs.fab.conveyor import ConveyorBelt
import numpy as np
from typing import Dict, Tuple, Optional, Any, List
from os import path
import random


class FAB:
    def __init__(self, mode='day'):
        # architecture description
        # family of InConveyors labelled by their floors
        # 7 In-Conveyor C2
        # 2F : L2 L3(POD) / 3F : L1 L2 L3 / 6F : L1(POD) L2 L3
        # TODO : add operation logger which indicates the following information:
        # number of total lots arrived in each queue throughout the operation
        # number of carried lots in each queue throughout the operation & its ratio
        # visit count of rack master to each queue
        # occupancy ratio in each floor of rack master
        # other useful information/statistics of the operation
        self.rack = Rack()
        self.floors = [2, 3, 6]
        # label assigned to each conveyor belt
        self.labels = [2, 3, 5, 6, 7, 8, 9]
        self.capacities = [3, 2, 4, 2, 6, 3, 2]
        self.max_capacity = max(self.capacities)
        self.num_layers = len(self.labels)
        self.layers: Dict[int, ConveyorBelt] = {
            label: ConveyorBelt(capacity=self.capacities[i]) for i, label in enumerate(self.labels)
        }

        # label -> (floor, layer)
        self.label_decoder = {2: (2, 2), 3: (2, 3), 5: (3, 2), 6: (3, 3), 7: (6, 1), 8: (6, 2), 9: (6, 3)}
        self.label2floor = {label: self.label_decoder[label][0] for label in self.labels}
        self.pos2label = {0: (None, 2), 1: (2, 3), 2: (3, None), 3: (None, None), 4: (None, 5),
                          5: (5, 6), 6: (6, None), 7: (None, 7), 8: (7, 8), 9: (8, 9)}
        self.pos2floor = [2, 2, 2, 3, 3, 3, 3, 6, 6, 6]  # convert control point to floor

        # travel time of the rack master between two floors
        # d[i, j] = time consumed for rack master to move from (ctrl pt i) to (ctrl pt j)
        """
        self.distance_matrix = np.array([[0.,   2.5,  3.4,  6.87, 6.94, 7.01, 7.08, 8.85, 8.9,  8.96],
                                         [2.5,  0.,   2.5,  6.79, 6.87, 6.94, 7.01, 8.79, 8.85, 8.9],
                                         [3.4,  2.5,  0.,   6.72, 6.79, 6.87, 6.94, 8.73, 8.79, 8.85],
                                         [6.87, 6.79, 6.72, 0.,   2.5,  3.4,  4.02, 5.03, 5.11, 5.19],
                                         [6.94, 6.87, 6.79, 2.5,  0.,   2.5,  3.4,  4.95, 5.03, 5.11],
                                         [7.01, 6.94, 6.87, 3.4,  2.5,  0.,   2.5,  4.87, 4.95, 5.03],
                                         [7.08, 7.01, 6.94, 4.02, 3.4,  2.5,  0.,   4.80, 4.87, 4.95],
                                         [8.85, 8.79, 8.73, 5.03, 4.95, 4.87, 4.80, 0.,   2.5,  3.4],
                                         [8.9,  8.85, 8.79, 5.11, 5.03, 4.95, 4.87, 2.5,  0.,   2.5],
                                         [8.96, 8.9,  8.85, 5.19, 5.11, 5.03, 4.95, 3.4,  2.5,  0.]])
        """

        self.distance_matrix = np.array([[0.,   2.5,  3.4,  4.95, 5.03, 5.11, 5.19, 8.85, 8.9,  8.96],
                                         [2.5,  0.,   2.5,  4.87, 4.95, 5.03, 5.11, 8.79, 8.85, 8.9],
                                         [3.4,  2.5,  0.,   4.8,  4.87, 4.95, 5.03, 8.73, 8.79, 8.85],
                                         [4.95, 4.87, 4.8,  0.,   2.5,  3.4,  4.02, 6.94, 7.01, 7.08],
                                         [5.03, 4.95, 4.87, 2.5,  0.,   2.5,  3.4,  6.87, 6.94, 7.01],
                                         [5.11, 5.03, 4.95, 3.4,  2.5,  0.,   2.5,  6.79, 6.87, 6.94],
                                         [5.19, 5.11, 5.03, 4.02, 3.4,  2.5,  0.,   6.72, 6.79, 6.87],
                                         [8.85, 8.79, 8.73, 6.94, 6.87, 6.79, 6.72, 0.,   2.5,  3.4],
                                         [8.9,  8.85, 8.79, 7.01, 6.94, 6.87, 6.79, 2.5,  0.,   2.5],
                                         [8.96, 8.9,  8.85, 7.08, 7.01, 6.94, 6.87, 3.4,  2.5,  0.]])

        self.rack_pos = None
        self.mode = mode
        self.data_cmd = None
        self.data_from = None
        self.data_to = None
        self.num_data = None
        self.num_added = None
        self.end = None
        self.t = None
        self.t_unit = 6
        self.visit_count = None
        # statistics
        self.num_carried = None
        self.load_two = None
        self.unload_two = None
        self.load_sequential = None
        self.total_amount = None
        self.arrival = None

    def reset(self, mode=None):
        self.rack.reset()
        if mode is None:
            self.arrival = True
            self.rack_pos = np.random.randint(low=0, high=10)
            for conveyor in self.layers.values():
                conveyor.reset()
        elif mode == 'test2':
            self.arrival = False
            for conveyor in self.layers.values():
                conveyor.reset()
            self.rack_pos = 5
            for label, conveyor in self.layers.items():
                from_floor = self.label2floor[label]
                if label == 2:
                    for _ in range(2):
                        wafer = Wafer(cmd_t=0., origin=from_floor, destination=3)
                        conveyor.push(wafer)
                elif label == 5:
                    wafer = Wafer(cmd_t=0., origin=from_floor, destination=2)
                    conveyor.push(wafer)
                elif label == 6:
                    wafer = Wafer(cmd_t=0., origin=from_floor, destination=6)
                    conveyor.push(wafer)
                elif label == 8:
                    wafer = Wafer(cmd_t=0., origin=from_floor, destination=2)
                    conveyor.push(wafer)
        else:
            self.arrival = False
            for conveyor in self.layers.values():
                conveyor.reset()
            self.rack_pos = 5
            for label, conveyor in self.layers.items():
                if label in [2, 5, 6, 8, 9]:
                    from_floor = self.label2floor[label]
                    for _ in range(2):
                        if from_floor == 2:
                            to_floor = random.choice([3, 6])
                        elif from_floor == 3:
                            to_floor = random.choice([6, 2])
                        else:
                            to_floor = random.choice([2, 3])

                        wafer = Wafer(cmd_t=0., origin=from_floor, destination=to_floor)
                        conveyor.push(wafer)
        self.load_arrival_data()
        self.end = 0
        self.t = 0.

        # FAB statistics
        self.num_carried = 0
        self.visit_count = np.zeros(10, dtype=int)
        self.total_amount = np.zeros(7, dtype=int)
        self.load_two = 0
        self.unload_two = 0
        self.load_sequential = 0

    def sim(self, operation: Optional[Tuple[int, int, int]]) -> Dict[str, Any]:
        self.visit_count[self.rack_pos] += 1
        if operation is None:
            # no rack operation
            # action staying at the position is not chosen unless all facilities are empty
            # it is free to set its operation time to be small
            operation_time = 8.
        else:
            pos, low_up, load_unload = operation
            # operation : move to the desired position -> load or unload
            # travel_t + loading/unloading_t
            operation_time = self.distance_matrix[pos, self.rack_pos] + 3.
            self.rack_pos = pos

            if low_up == 0:
                if load_unload == 0:
                    self.load_lower()
                    if self.rack.is_upper_loaded:
                        self.load_sequential += 1
                elif load_unload == 1:
                    assert self.rack.destination[0] == self.pos2floor[self.rack_pos]
                    self.num_carried += 1
                    self.rack.release_lower_fork()
            elif low_up == 1:
                if load_unload == 0:
                    self.load_upper()
                    if self.rack.is_lower_loaded:
                        self.load_sequential += 1
                elif load_unload == 1:
                    assert self.rack.destination[1] == self.pos2floor[self.rack_pos]
                    self.num_carried += 1
                    self.rack.release_upper_fork()
            elif low_up == 2:
                if load_unload == 0:
                    self.load_lower(), self.load_upper()
                    self.load_two += 1
                elif load_unload == 1:
                    assert self.rack.destination[0] == self.pos2floor[self.rack_pos]
                    assert self.rack.destination[1] == self.pos2floor[self.rack_pos]
                    self.rack.release_lower_fork(), self.rack.release_upper_fork()
                    self.num_carried += 2
                    self.unload_two += 1
        # simulation of lots arrival
        # performed by reading the simulation data
        if self.arrival:
            done = self.sim_arrival(dt=operation_time)
        else:
            self.t += operation_time
        info = {
                'dt': operation_time / self.t_unit,
                'elapsed_time': self.elapsed_time / self.t_unit,
                }
        return info

    def sim_arrival(self, dt: float):
        # read data for simulation
        assert dt > 0.
        begin = self.end
        next_t = self.t + dt
        while self.end < self.num_data:
            if self.t < self.data_cmd[self.end] <= next_t:
                self.end += 1
            else:
                break
        self.num_added = self.end - begin
        for i in range(begin, self.end):
            # wafer generation from data
            wafer = Wafer(cmd_t=self.data_cmd[i], origin=self.data_from[i], destination=self.data_to[i])
            # arrived lots are randomly distributed into several layers
            if self.data_from[i] == 2:
                self.layers[2].push(wafer)
                self.total_amount[0] += 1
            elif self.data_from[i] == 3:
                coin = np.random.rand()
                if coin < .5:
                    self.layers[5].push(wafer)
                    self.total_amount[2] += 1
                else:
                    self.layers[6].push(wafer)
                    self.total_amount[3] += 1

            elif self.data_from[i] == 6:
                coin = np.random.rand()
                if coin < .5:
                    self.layers[8].push(wafer)
                    self.total_amount[5] += 1
                else:
                    self.layers[9].push(wafer)
                    self.total_amount[6] += 1
        if self.end == self.num_data:
            done = True
        else:
            done = False
        self.t = next_t
        return done

    def load_arrival_data(self):
        scenario = np.random.randint(low=0, high=200)
        if self.mode == 'day':
            # poisson
            dir_path = 'assets/day/scenario{}/'.format(scenario)
        elif self.mode == 'day_uniform':
            # uniform
            dir_path = 'assets/day_uniform/scenario{}/'.format(scenario)
        else:
            dir_path = 'assets/half_hr/scenario{}/'.format(scenario)
        self.data_cmd = np.load(path.join(path.dirname(__file__), dir_path + "data_cmd.npy"))
        self.data_from = np.load(path.join(path.dirname(__file__), dir_path + "data_from.npy"))
        self.data_to = np.load(path.join(path.dirname(__file__), dir_path + "data_to.npy"))
        self.num_data = self.data_cmd.shape[0]

    def render(self):
        # TODO
        return

    def load_lower(self):
        target_label = self.pos2label[self.rack_pos][0]
        assert target_label is not None
        target_conveyor = self.layers[target_label]
        assert not target_conveyor.is_empty
        self.rack.load_lower(target_conveyor.pop())
        return

    def load_upper(self):
        target_label = self.pos2label[self.rack_pos][1]
        assert target_label is not None
        target_conveyor = self.layers[target_label]
        assert not target_conveyor.is_empty
        self.rack.load_upper(target_conveyor.pop())
        return

    @property
    def operation_log(self):
        info = {
            'carried': self.num_carried,
            'waiting_quantity': self.waiting_quantity,
            'visit_count': self.visit_count,
            'load_two': self.load_two,
            'unload_two': self.unload_two,
            'load_sequential': self.load_sequential,
            'total': self.total_amount,
            'pod_total': self.total_amount[1] + self.total_amount[4]
        }
        return info

    @property
    def waiting_time(self):
        wt = np.zeros(self.num_layers)
        for i, conveyors in enumerate(self.layers.values()):
            wt[i] = 0. if conveyors.is_empty else self.t - conveyors.cmd_time
        return wt

    @property
    def destination(self) -> List[int]:
        return [conveyor.destination for conveyor in self.layers.values()]

    @property
    def rack_destination(self) -> Tuple[int, int]:
        return self.rack.destination

    @property
    def is_pod_loaded(self):
        return self.rack.is_pod_loaded

    @property
    def travel_distance(self):
        # TODO : what if the queue is empty?
        return np.array([self.layers[i].destination - self.label2floor[i] for i in self.layers]) / 4.

    @property
    def elapsed_time(self):
        return self.t

    @property
    def num_lots(self):
        return [(label, conveyor.QUEUE_LEN) for label, conveyor in self.layers.items()]

    @property
    def waiting_quantity(self) -> List[int]:
        return [conveyor.QUEUE_LEN for conveyor in self.layers.values()]
