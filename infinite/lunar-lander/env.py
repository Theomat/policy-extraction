from typing import Callable, List
import numpy as np
from stable_baselines3 import DQN

from polext import Predicate
import torch
import gym

bins = 20
states_arrays = [
    ("x", np.linspace(-1., +1., num=bins, endpoint=False)),
    ("y", np.linspace(-1.5, +1.5, num=bins, endpoint=False)),
    ("vx", np.linspace(-5, +5, num=bins, endpoint=False)),
    ("vy", np.linspace(-5, +5, num=bins, endpoint=False)),
    ("angle", np.linspace(-3.14, +3.14, num=bins, endpoint=False)),
    ("vangle", np.linspace(-5, 5, num=bins, endpoint=False)),
    ("left leg on ground", [1]),
    ("right leg on ground", [1]),

]


make_env = lambda: gym.make("LunarLander-v2")
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
    model = DQN(
        "MlpPolicy", make_env(), policy_kwargs={"net_arch": [256, 256]}
    )
    model = model.load(path)

    def f(observation: np.ndarray) -> List[float]:
        observation = torch.tensor(observation, device=model.device).unsqueeze_(0)
        # print(observation)
        # assert False
        with torch.no_grad():
            q_values = model.q_net(observation)[0]
        return [x.item() for x in q_values]

    return f