from gym_lifter.envs.wafer import Wafer
from typing import List


class ConveyorBelt:
    def __init__(self):
        self._MAX_CAPACITY = 200
        self._QUEUE_LEN = 0

        self._QUEUE: List[Wafer] = []   # all elements of the queue must be Wafer objects
        return

    def push(self, wafer):
        if self._QUEUE_LEN == self._MAX_CAPACITY:
            # if max capacity is reached, then we cannot let a new wafer in
            return
        else:
            self._QUEUE.append(wafer)
            self._QUEUE_LEN += 1
        return

    def pop(self):
        assert self._QUEUE_LEN > 0
        wafer = self._QUEUE.pop(0)
        self._QUEUE_LEN -= 1
        return wafer

    @property
    def cmd_time(self):
        return self._QUEUE[0].cmd_time if self._QUEUE_LEN > 0 else 0.

    @property
    def destination(self):
        return self._QUEUE[0].destination if self._QUEUE_LEN > 0 else 0

    @property
    def is_empty(self):
        return True if self._QUEUE_LEN == 0 else False

    def __len__(self):
        return self._QUEUE_LEN

