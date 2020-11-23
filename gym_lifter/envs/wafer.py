import typing


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

    @property
    def is_pod(self) -> bool:
        # TODO : to be added when the model distinguishes between normal FOUPS & special PODS
        return False
