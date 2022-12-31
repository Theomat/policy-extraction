from typing import Callable, List
import numpy as np
from stable_baselines3 import DQN

from polext import Predicate
import torch
import gym

bins = 40

states_arrays = [
    ("x", np.linspace(-1.0, +1.0, num=bins, endpoint=False)),
    ("y", np.linspace(-0.2, +2.0, num=bins, endpoint=False)),
    ("vx", np.linspace(-1, +1, num=bins, endpoint=False)),
    ("vy", np.linspace(-1, +1, num=bins, endpoint=False)),
    ("angle", np.linspace(-3.14 / 2, +3.14 / 2, num=bins, endpoint=False)),
    ("vangle", np.linspace(-3.14, 3.14, num=bins, endpoint=False)),
    ("left leg on ground", [1]),
    ("right leg on ground", [1]),
]


make_env = lambda: gym.make("LunarLander-v2")
env = make_env()


def pred(i: int, val: float):
    def f(s) -> bool:
        return s[i] >= val

    return f


predicates = []
for i, (name, array) in enumerate(states_arrays):
    for el in array:
        predicates.append(Predicate(f"{name} >= {el:.2e}", pred(i, el)))


def idx_from_name(name: str) -> int:
    for i, (idx_name, _) in enumerate(states_arrays):
        if idx_name == name:
            return i
    return 999999


def make_pair_pred(na: str, nb: str, nber: float):
    ai = idx_from_name(na)
    bi = idx_from_name(nb)

    def lower(s):
        return s[ai] + s[bi] <= -nber

    def higher(s):
        return s[ai] + s[bi] >= nber

    return lower, higher


for na, nb in [("angle", "vangle"), ("x", "vx"), ("y", "vy")]:
    for nbr in [0, 0.05, 0.1]:
        lower, higher = make_pair_pred(na, nb, nbr)
        predicates.append(Predicate(f"{na} + {nb} >= {nbr}", higher))
        predicates.append(Predicate(f"{na} + {nb} <= {nbr}", lower))

predicates.append(Predicate("has ground contact", lambda s: s[-1] + s[-2] >= 1))


def center_angle(s):
    angle_targ = s[0] * 0.5 + s[2] * 1.0  # angle should point towards center
    if angle_targ > 0.4:
        angle_targ = 0.4  # more than 0.4 radians (22 degrees) is bad
    if angle_targ < -0.4:
        angle_targ = -0.4
    return angle_targ


predicates.append(
    Predicate(
        "(center - angle)/2 - vangle <= .05",
        lambda s: (center_angle(s) - s[4]) / 2 - s[5] <= 0.05,
    )
)
predicates.append(
    Predicate(
        "(center - angle)/2 - vangle >= .05",
        lambda s: (center_angle(s) - s[4]) / 2 - s[5] >= 0.05,
    )
)


def Q_builder(path: str) -> Callable[[np.ndarray], List[float]]:
    model = DQN("MlpPolicy", make_env(), policy_kwargs={"net_arch": [256, 256]})
    model = model.load(path, buffer_size=0)

    def f(observation: np.ndarray) -> List[float]:
        obs = torch.tensor(observation, device=model.device)
        batched = len(obs.shape) == 2
        if not batched:
            obs.unsqueeze_(0)
        # print(observation)
        # assert False
        with torch.no_grad():
            q_values = model.q_net(obs).cpu().numpy()
        if batched:
            return q_values
        else:
            return q_values[0]

    return f
