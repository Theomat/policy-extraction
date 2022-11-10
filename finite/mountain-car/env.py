from typing import Callable, List, Tuple
import numpy as np
from stable_baselines3 import DQN

from polext import Predicate
import torch
import gym
from gym.spaces import MultiDiscrete

bins = 15
velocity_state_array = np.linspace(-1.5, +1.5, num=bins, endpoint=False)
position_state_array = np.linspace(-1.2, +0.5, num=bins, endpoint=False)


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
    for j, el in enumerate(array):
        predicates.append(Predicate(f"{name} >= {el:.2e}", pred(i, j)))


def Q_builder(path: str) -> Callable[[Tuple[int, int]], List[float]]:
    model = DQN(
        "MlpPolicy", gym.make("MountainCar-v0"), policy_kwargs={"net_arch": [256, 256]}
    )
    model = model.load(path)

    def f(state: Tuple[int, int]) -> List[float]:
        i, j = state
        velocity = (
            velocity_state_array[i - 1]
            + (velocity_state_array[i] if i < bins else velocity_state_array[i - 1])
        ) / 2
        pos = (
            position_state_array[j - 1]
            + (position_state_array[j] if j < bins else position_state_array[j - 1])
        ) / 2
        float_state = (pos, velocity)
        observation = np.array(float_state).reshape(
            (-1,) + model.observation_space.shape
        )

        observation = torch.tensor(observation, device=model.device)
        with torch.no_grad():
            q_values = model.q_net(observation)[0]
        return [x.item() for x in q_values]

    return f
