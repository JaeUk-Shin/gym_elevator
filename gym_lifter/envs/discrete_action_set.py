import numpy as np
from typing import List


def available_actions_no_wt(state: np.ndarray) -> List[int]:
    # 0 : go up
    # 1 : go down
    # 2 : load lower
    # 3 : load upper
    # 4 : load both
    # 5 : unload lower
    # 6 : unload upper
    # 7 : unload both
    rack_pos = int(round(9. * state[0]))
    actions = []
    lower_floor = int(round(6. * state[2]))
    upper_floor = int(round(6. * state[3]))

    if lower_floor == 0:
        # lower fork is empty
        lower_occupied = False
    else:
        lower_occupied = True

    if upper_floor == 0:
        # upper fork is empty
        upper_occupied = False
    else:
        upper_occupied = True
    is_pod = bool(round(state[1]))
    wq = state[-7:]     # waiting quantity

    if rack_pos > 0:
        actions.append(1)
    if rack_pos < 9:
        actions.append(0)

    if is_pod:
        pass
    else:
        if lower_occupied:
            coincide: bool = (lower_floor == 2 and rack_pos == 1)\
                                or (lower_floor == 3 and rack_pos in [5, 6])\
                                or (lower_floor == 6 and rack_pos == 9)
            if coincide:
                actions.append(5)

        else:

            # can load at lower fork
            rack_layer = [0, 2, 3, 5]   # control points
            ctrl_pts = [1, 5, 6, 9]   # control points
            for i in range(4):
                if rack_pos == ctrl_pts[i] and wq[rack_layer[i]] > 0:
                    actions.append(2)
                    break

        if upper_occupied:
            coincide: bool = (upper_floor == 2 and rack_pos == 0) \
                             or (upper_floor == 3 and rack_pos in [4, 5]) \
                             or (upper_floor == 6 and rack_pos in [8, 9])
            if coincide:
                actions.append(6)

        else:
            # can load at upper fork
            rack_layer = [0, 2, 3, 5, 6]
            ctrl_pts = [0, 4, 5, 8, 9]

            for i in range(4):
                if rack_pos == ctrl_pts[i] and wq[rack_layer[i]] > 0:
                    actions.append(3)
                    break

        if not (upper_occupied or lower_occupied):
            coincide = (wq[2] > 0 and wq[3] > 0 and rack_pos == 5) or (wq[5] > 0 and wq[6] > 0 and rack_pos == 9)
            if coincide:
                actions.append(4)

            # TODO : add POD

        if upper_occupied and lower_occupied:
            coincide = (upper_floor == 3 and lower_floor == 3 and rack_pos == 5) or (upper_floor == 6 and lower_floor == 6 and rack_pos == 9)
            if coincide:
                actions.append(7)
    # print(actions)
    return actions
