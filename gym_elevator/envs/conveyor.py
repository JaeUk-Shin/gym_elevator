class ConveyorBelt:
    def __init__(self):
        self._MAX_CAPACITY = 200
        self._QUEUE_LEN = 0

        self._QUEUE = []
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
        if self._QUEUE_LEN == 0:
            return 0
        else:
            return self._QUEUE[0].cmd_time

    @property
    def destination(self):
        assert self._QUEUE_LEN > 0
        return self._QUEUE[0].destination

    def __len__(self):
        return self._QUEUE_LEN


class Wafer:
    def __init__(self, cmd_t, origin, destination):
        self._CMD_TIME = cmd_t
        self._FROM = origin
        self._TO = destination

    @property
    def cmd_time(self):
        return self._CMD_TIME

    @property
    def origin(self):
        return self._FROM

    @property
    def destination(self):
        return self._TO
