from gym.envs.registration import register


register(
    id='Elevator-v0',
    entry_point='gym_elevator.envs:ElevatorEnv',
    max_episode_steps=99,
)

