from typing import Callable
import numpy as np
from stable_baselines3 import DQN

from polext import Predicate
import torch
import gym
from gym.spaces import MultiDiscrete

bins = 60

states_arrays = [
    ("position", np.linspace(-1.2, +0.5, num=bins, endpoint=False)),
    ("speed", np.linspace(-0.07, +0.07, num=bins, endpoint=False)),
]


class DiscreteWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = MultiDiscrete(
            [bins for _ in range(len(states_arrays))]
        )

    def observation(self, obs):
        return np.asarray(
            [
                np.digitize(obs[i], states_arrays[i][1])
                for i in range(len(states_arrays))
            ]
        )


make_env = lambda: DiscreteWrapper(gym.make("MountainCar-v0"))
env = make_env()

states = [np.asarray((i, j)) for i in range(bins) for j in range(bins)]


def pred(i: int, val: float):
    def f(s) -> bool:
        return s[i] > val

    return f


predicates = []
for i, (name, array) in enumerate(states_arrays):
    for j, el in enumerate(array):
        predicates.append(Predicate(f"{name} >= {el:.2e}", pred(i, j)))


def real(array: np.ndarray, i: int) -> float:
    return (array[i - 1] + (array[i] if i < bins else array[i - 1])) / 2


def Q_builder(path: str) -> Callable[[np.ndarray], np.ndarray]:
    model = DQN(
        "MlpPolicy", gym.make("MountainCar-v0"), policy_kwargs={"net_arch": [256, 256]}
    )
    model = model.load(path)

    def f(state: np.ndarray) -> np.ndarray:
        batched = len(state.shape) == 2

        if batched:
            float_state = [
                [
                    real(states_arrays[i][1], state[j, i])
                    for i in range(len(states_arrays))
                ]
                for j in range(state.shape[0])
            ]
        else:
            float_state = tuple(
                real(states_arrays[i][1], state[i]) for i in range(len(states_arrays))
            )
        observation = np.array(float_state).reshape(
            (-1,) + model.observation_space.shape
        )

        observation = torch.tensor(observation, device=model.device)
        with torch.no_grad():
            q_values = model.q_net(observation).cpu().numpy()
        if batched:
            return q_values
        else:
            return q_values[0]

    return f
