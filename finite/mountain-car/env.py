import numpy as np

import gym
from gym.spaces import MultiDiscrete

bins = 12
velocity_state_array = np.linspace(-1.5, +1.5, num=bins - 1, endpoint=False)
position_state_array = np.linspace(-1.2, +0.5, num=bins - 1, endpoint=False)


class DiscreteWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = MultiDiscrete([bins, bins])

    def observation(self, obs):
        return (
            np.digitize(obs[1], velocity_state_array),
            np.digitize(obs[0], position_state_array),
        )


make_env = lambda: DiscreteWrapper(gym.make("MountainCar-v0"))
env = make_env()

states = [(i, j) for i in range(bins) for j in range(bins)]
