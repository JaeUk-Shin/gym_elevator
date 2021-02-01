class Wafer:
    def __init__(self, cmd_t, origin, destination, pod=False):
        self.CMD_T = cmd_t
        self.FROM = origin
        self.TO = destination
        self.IS_POD = pod

    @property
    def cmd_time(self):
        return self.CMD_T

    @property
    def origin(self):
        return self.FROM

    @property
    def destination(self):
        return self.TO

    @property
    def is_pod(self) -> bool:
        # TODO : to be added when the model distinguishes between normal FOUPS & special PODS
        return self.IS_POD
