from gym.envs.registration import register


register(
    id='Lifter-v0',
    entry_point='gym_lifter.envs:LifterEnv',
    max_episode_steps=100,
)

