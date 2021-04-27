from gym.envs.registration import register


register(
    id='Lifter-v0',
    entry_point='gym_lifter.envs:LifterEnv',
    max_episode_steps=5000,
)

register(
    id='LifterAutomod-v0',
    entry_point='gym_lifter.envs:AutomodLifterEnv',
    max_episode_steps=5000,
)

register(
    id='DiscreteLifter-v0',
    entry_point='gym_lifter.envs:DiscreteLifterEnv',
    max_episode_steps=28800,
)
