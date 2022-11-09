from typing import Callable, List, Tuple
import numpy as np
from stable_baselines3 import DQN

from polext import Predicate
import torch
import gym
from gym.spaces import MultiDiscrete

bins = 15
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


def pred(i: int, val: float):
    def f(s) -> bool:
        return s[i] > val

    return f


predicates = []
for i, array, name in [
    (0, velocity_state_array, "speed"),
    (1, position_state_array, "pos"),
]:
    for el in array[1:]:
        predicates.append(Predicate(f"{name} > {el:.2e}", pred(i, el)))


def Q_builder(path: str) -> Callable[[Tuple[int, int]], List[float]]:
    model = DQN("MlpPolicy", make_env()).load(path)

    def f(state: Tuple[int, int]) -> List[float]:
        float_state = (velocity_state_array[state[0]], position_state_array[state[1]])
        observation = np.array(float_state).reshape(
            (-1,) + model.observation_space.shape
        )

        observation = torch.tensor(observation, device=model.device)
        with torch.no_grad():
            q_values = model.q_net(observation)[0]
        return [x.item() for x in q_values]

    return f
