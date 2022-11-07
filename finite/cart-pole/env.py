import numpy as np

import gym
from gym.spaces import MultiDiscrete

bins = 7
velocity_state_array = np.linspace(-3, +3, num=bins - 1, endpoint=False)
position_state_array = np.linspace(-2.4, +2.4, num=bins - 1, endpoint=False)
angle_state_array = np.linspace(-0.2095, +0.2095, num=bins - 1, endpoint=False)
angle_velocity_state_array = np.linspace(-2.0, +2.0, num=bins - 1, endpoint=False)


class DiscreteWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = MultiDiscrete([bins, bins, bins, bins])

    def observation(self, obs):
        return (
            np.digitize(obs[0], angle_velocity_state_array),
            np.digitize(obs[1], angle_state_array),
            np.digitize(obs[2], velocity_state_array),
            np.digitize(obs[3], position_state_array),
        )


make_env = lambda: DiscreteWrapper(gym.make("CartPole-v1"))
env = make_env()

states = [
    (i, j, k, l)
    for l in angle_velocity_state_array
    for k in angle_state_array
    for i in velocity_state_array
    for j in position_state_array
]
