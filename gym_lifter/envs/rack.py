from typing import Optional, Tuple
from gym_lifter.envs.wafer import Wafer


class Rack:
    def __init__(self):
        self._UPPER_FORK: Optional[Wafer] = None
        self._LOWER_FORK: Optional[Wafer] = None

        self.is_upper_loaded: bool = False
        self.is_lower_loaded: bool = False
        self.is_pod_loaded: bool = False

    def release_upper_fork(self):
        if not self.is_upper_loaded:
            return
        else:
            self._UPPER_FORK = None
            self.is_upper_loaded = False
        return

    def release_lower_fork(self):
        if not self.is_lower_loaded:
            return
        else:
            self._LOWER_FORK = None
            self.is_lower_loaded = False
            if self.is_pod_loaded:
                self.is_pod_loaded = False
        return

    def load_upper(self, wafer: Wafer):
        assert ~self.is_upper_loaded
        self._UPPER_FORK = wafer
        self.is_upper_loaded = True
        return

    def load_lower(self, wafer: Wafer):
        assert ~self.is_lower_loaded
        self._LOWER_FORK = wafer
        self.is_lower_loaded = True
        return

    def load_pod(self, wafer: Wafer):
        assert wafer.is_pod and ~self.is_upper_loaded and ~self.is_lower_loaded
        self._LOWER_FORK = wafer
        self.is_lower_loaded = True
        self.is_pod_loaded = True

    @property
    def destination(self) -> Tuple[int, int]:
        destination1 = self._LOWER_FORK.destination if self.is_lower_loaded else 0
        destination2 = self._UPPER_FORK.destination if self.is_upper_loaded else 0
        return destination1, destination2

