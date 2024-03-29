from typing import Callable, List
import numpy as np
from stable_baselines3 import DQN

from polext import Predicate
import torch
import gym

bins = 10
states_arrays = [
    ("cos(theta1)", np.linspace(-1.0, +1.0, num=bins, endpoint=False)),
    ("sin(theta1)", np.linspace(-1.0, +1.0, num=bins, endpoint=False)),
    ("cos(theta2)", np.linspace(-1.0, +1.0, num=bins, endpoint=False)),
    ("sin(theta2)", np.linspace(-1.0, +1.0, num=bins, endpoint=False)),
    ("moment of theta1", np.linspace(-12.567, +12.567, num=bins, endpoint=False)),
    ("moment of theta2", np.linspace(-28.274, 28.274, num=bins, endpoint=False)),
]


make_env = lambda: gym.make("Acrobot-v1")
env = make_env()


def pred(i: int, val: float):
    def f(s) -> bool:
        return s[i] > val

    return f


predicates = []
for i, (name, array) in enumerate(states_arrays):
    for el in array:
        predicates.append(Predicate(f"{name} >= {el:.2e}", pred(i, el)))


def Q_builder(path: str) -> Callable[[np.ndarray], List[float]]:
    model = DQN("MlpPolicy", make_env(), policy_kwargs={"net_arch": [256, 256]})
    model = model.load(path, buffer_size=0)

    def f(observation: np.ndarray) -> np.ndarray:
        observation = torch.tensor(observation, device=model.device)
        batched = len(observation.shape) == 2
        if not batched:
            observation.unsqueeze_(0)
        # print(observation)
        # assert False
        with torch.no_grad():
            q_values = model.q_net(observation).cpu().numpy()
        if batched:
            return q_values
        else:
            return q_values[0]

    return f
