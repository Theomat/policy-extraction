from typing import Callable, List, Tuple
import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env.vec_frame_stack import VecFrameStack
from stable_baselines3.common.vec_env.base_vec_env import (
    VecEnvWrapper,
    VecEnv,
    VecEnvStepReturn,
    VecEnvObs,
)
from stable_baselines3.common.env_util import make_atari_env

from polext import Predicate

import torch


class VecActionWrapper(VecEnvWrapper):
    """
    Vectorize the action

    :param env: the environment to wrap
    """

    def __init__(self, env: VecEnv) -> None:
        super().__init__(env)

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        obs, r, done, info = self.venv.step([action])
        return obs, r[0], done[0], info

    def step_wait(self) -> VecEnvStepReturn:
        return self.venv.step_wait()

    def reset(self) -> VecEnvObs:
        obs = self.venv.reset()
        return obs


make_env = lambda: VecActionWrapper(
    VecFrameStack(make_atari_env("PongNoFrameskip-v4"), 4)
)


def Q_builder(path: str) -> Callable[[np.ndarray], np.ndarray]:
    model = DQN("CnnPolicy", make_env(), buffer_size=0)
    model = model.load(path)

    def f(observation: np.ndarray) -> np.ndarray:
        observation = (
            torch.tensor(observation, device=model.device).swapdims(1, 3).swapdims(2, 3)
        )

        with torch.no_grad():
            q_values = model.q_net(observation)[0]
        return q_values.numpy()

    return f


def listify(x):
    if isinstance(x, tuple):
        return [listify(y) for y in x]
    return x


def ready(obs):
    if isinstance(obs, tuple):
        obs = np.asarray(listify(obs))
    if isinstance(obs, np.ndarray):
        nobs = np.asarray(obs)
        if len(nobs.shape) == 4:
            nobs = nobs[0]
        return nobs
    return obs


predicates = [
    Predicate(
        "my paddle y < ball y",
        lambda obs: get_paddle_y(MY_PADDLE, ready(obs)) < get_ball_pos(ready(obs))[1],
    ),
]


MY_PADDLE = 73
ENEMY_PADDLE = 10
PADDLE_HEIGHT = 7


def get_paddle_y(paddle: int, img: np.ndarray) -> int:
    return np.argmax(img[14:76, paddle] != 87)


def get_ball_pos(img: np.ndarray, index: int = 3) -> Tuple[int, int]:
    tmp = img[14:76, ENEMY_PADDLE + 1 : MY_PADDLE, index] != 87
    x = np.max(np.argmax(tmp, axis=0, keepdims=True))
    return x, np.argmax(tmp[x])


def get_ball_speed(img: np.ndarray) -> Tuple[int, int]:
    dx, dy = get_ball_pos(img, 3)
    sx, sy = get_ball_pos(img, 0)
    return dx - sx, dy - sy


def make_pred(val: int, getter: Callable):
    def f(obs):
        return val <= getter(ready(obs))

    return f


for i in range(ENEMY_PADDLE + 1, MY_PADDLE, 10):
    predicates.append(
        Predicate(f"ball x <= {i}", make_pred(i, lambda obs: get_ball_pos(obs)[0]))
    )

for i in range(0, 62, 10):
    predicates.append(
        Predicate(
            f"my paddle y <= {i}",
            make_pred(i, lambda obs: get_paddle_y(MY_PADDLE, obs)),
        )
    )
    predicates.append(
        Predicate(
            f"enemy paddle y <= {i}",
            make_pred(i, lambda obs: get_paddle_y(ENEMY_PADDLE, obs)),
        )
    )
    predicates.append(
        Predicate(f"ball y <= {i}", make_pred(i, lambda obs: get_ball_pos(obs)[1]))
    )

for i in range(7):
    predicates.append(
        Predicate(f"ball vx <= {i}", make_pred(i, lambda obs: get_ball_speed(obs)[0]))
    )
    predicates.append(
        Predicate(f"ball vy <= {i}", make_pred(i, lambda obs: get_ball_speed(obs)[1]))
    )

# if __name__ == "__main__":
#     env = make_env()
#     obs = env.reset()
#     for i in range(400):
#         obs = env.step([0])[0].squeeze(0)
#     print(obs.shape)
#     for predicate in predicates:
#         print(predicate.name, predicate(obs))
#     from matplotlib import pyplot as plt
#     plt.figure()
#     for i in range(4):
#         plt.subplot(1, 4, 1 + i)
#         plt.imshow(obs[14:76, ENEMY_PADDLE:MY_PADDLE + 1, i])
#     plt.show()
# print("my paddle:", get_paddle_y(MY_PADDLE, obs))
# print("enemy paddle:", get_paddle_y(ENEMY_PADDLE, obs))
# print("ball pos:", get_ball(obs))
# py =
# print(py)
