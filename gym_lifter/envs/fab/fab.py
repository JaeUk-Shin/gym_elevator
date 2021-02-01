from gym_lifter.envs.fab.wafer import Wafer
from gym_lifter.envs.fab.rack import Rack
from gym_lifter.envs.fab.conveyor import ConveyorBelt
import numpy as np
from typing import Dict, Tuple, Optional, Any
from os import path


class FAB:
    def __init__(self):

        # architecture description
        # family of InConveyors labelled by their floors
        # 7 In-Conveyor C2
        # 2F : L2 L3(POD) / 3F : L1 L2 L3 / 6F : L1(POD) L2 L3
        self.rack = Rack()
        self.floors = [2, 3, 6]
        # label assigned to each conveyor belt
        self.labels = [2, 3, 5, 6, 7, 8, 9]
        self.num_layers = len(self.labels)
        self.layers: Dict[int, ConveyorBelt] = {label: ConveyorBelt() for label in self.labels}

        # label -> (floor, layer)
        self.label_decoder = {2: (2, 2), 3: (2, 3), 5: (3, 2), 6: (3, 3), 7: (6, 1), 8: (6, 2), 9: (6, 3)}
        self.label2floor = {label: self.label_decoder[label][0] for label in self.labels}
        self.pos2label = {0: (None, 2), 1: (2, 3), 2: (3, None), 3: (None, None), 4: (None, 5),
                          5: (5, 6), 6: (6, None), 7: (None, 7), 8: (7, 8), 9: (8, 9)}
        self.pos2floor = [2, 2, 2, 3, 3, 3, 3, 6, 6, 6]  # convert control point to floor

        self.rack_pos = None

        self.data_cmd = None
        self.data_from = None
        self.data_to = None
        self.num_data = None
        self.num_added = None
        self.end = None
        self.t = None
        self.t_unit = 5.5

        # statistics
        self.num_carried = None

    def reset(self):
        self.rack.reset()
        self.rack_pos = np.random.randint(low=0, high=10)
        for conveyor in self.layers.values():
            conveyor.reset()
        self.load_arrival_data()

        self.num_carried = 0
        self.end = 0
        self.t = 0.

    def sim(self, operation: Optional[Tuple[int, int, int]]) -> Dict[str, Any]:
        if operation is None:
            # no rack operation
            operation_time = 2.5
        else:
            pos, low_up, load_unload = operation
            # operation : move to the desired position -> load or unload
            operation_time = np.abs(pos - self.rack_pos) * 2.5 + 3.  # travel_t + loading/unloading_t
            self.rack_pos = pos

            if low_up == 0:
                if load_unload == 0:
                    self.load_lower()
                elif load_unload == 1:
                    assert self.rack.destination[0] == self.pos2floor[self.rack_pos]
                    self.num_carried += 1
                    self.rack.release_lower_fork()
            elif low_up == 1:
                if load_unload == 0:
                    self.load_upper()
                elif load_unload == 1:
                    assert self.rack.destination[1] == self.pos2floor[self.rack_pos]
                    self.num_carried += 1
                    self.rack.release_upper_fork()
            elif low_up == 2:
                if load_unload == 0:
                    self.load_lower(), self.load_upper()
                elif load_unload == 1:
                    assert self.rack.destination[0] == self.pos2floor[self.rack_pos]
                    assert self.rack.destination[1] == self.pos2floor[self.rack_pos]
                    self.rack.release_lower_fork(), self.rack.release_upper_fork()
                    self.num_carried += 2
        self.sim_arrival(dt=operation_time)
        info = {'dt': operation_time / self.t_unit,
                'carried': self.num_carried,
                'elapsed time': self.elapsed_time / self.t_unit}
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

            elif self.data_from[i] == 3:
                coin = np.random.rand()
                if coin < .5:
                    self.layers[5].push(wafer)
                else:
                    self.layers[6].push(wafer)

            elif self.data_from[i] == 6:
                coin = np.random.rand()
                if coin < .5:
                    self.layers[8].push(wafer)
                else:
                    self.layers[9].push(wafer)

        self.t = next_t
        return

    def load_arrival_data(self):
        scenario = np.random.randint(low=0, high=200)
        dir_path = 'assets/scenario{}/'.format(scenario)
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
    def waiting_time(self):
        wt = np.zeros(self.num_layers)
        for i, conveyors in enumerate(self.layers.values()):
            wt[i] = 0. if conveyors.is_empty else self.t - conveyors.cmd_time
        return wt

    @property
    def destination(self):
        return np.array([conveyor.destination for conveyor in self.layers.values()])

    @property
    def rack_destination(self):
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
