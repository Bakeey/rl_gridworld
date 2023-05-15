from gymnasium.envs.registration import register

register(
    id="rl_gridworld/Dyson-v0",
    entry_point="rl_gridworld.envs:DysonEnv",
)
