from typing import Callable, List
import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.common.atari_wrappers import AtariWrapper

from polext import Predicate
import torch
import gym


make_env = lambda: AtariWrapper(gym.make("PongNoFrameskip-v4"))
env = make_env()


def Q_builder(path: str) -> Callable[[np.ndarray], List[float]]:
    model = DQN("CnnPolicy", make_env())
    model = model.load(path)

    def f(observation: np.ndarray) -> List[float]:
        observation = torch.tensor(observation, device=model.device).unsqueeze_(0)
        # print(observation)
        # assert False
        with torch.no_grad():
            q_values = model.q_net(observation)[0]
        return [x.item() for x in q_values]

    return f


# TODO: add predicates
