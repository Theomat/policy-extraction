from typing import Callable, List, Tuple
import numpy as np
from stable_baselines3 import DQN

import gym
from gym.spaces import MultiDiscrete
import torch

from polext import Predicate

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
            np.digitize(obs[3], angle_velocity_state_array),
            np.digitize(obs[2], angle_state_array),
            np.digitize(obs[1], velocity_state_array),
            np.digitize(obs[0], position_state_array),
        )


make_env = lambda: DiscreteWrapper(gym.make("CartPole-v1"))
env = make_env()

states = [
    (i, j, k, l)
    for l in range(bins)
    for k in range(bins)
    for i in range(bins)
    for j in range(bins)
]

predicates = [
    Predicate("angle_positive", lambda s: s[1] > 0),
    Predicate("angle_velocity_positive", lambda s: s[0] > 0),
    Predicate("velocity_positive", lambda s: s[2] > 0),
    Predicate("position_positive", lambda s: s[0] > 0),
]


def Q_builder(path: str) -> Callable[[Tuple[int, int, int, int]], List[float]]:
    model = DQN("MlpPolicy", make_env()).load(path)

    def f(state: Tuple[int, int, int, int]) -> List[float]:
        observation = np.array(state).reshape((-1,) + model.observation_space.shape)
        observation = torch.tensor(observation, device=model.device)
        with torch.no_grad():
            q_values = model.q_net(observation)
            print(q_values)
        return [x for x in q_values]

    return f
