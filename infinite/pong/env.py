from typing import Callable, Tuple
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

    def step_async(self, actions: np.ndarray) -> None:
        if not isinstance(actions, (np.ndarray, list, tuple)):
            return self.venv.step_async([actions])
        return self.venv.step_async(actions)

    def step_wait(self) -> VecEnvStepReturn:
        return self.venv.step_wait()

    def reset(self) -> VecEnvObs:
        return self.venv.reset()


make_env = lambda: VecActionWrapper(
    VecFrameStack(make_atari_env("PongNoFrameskip-v4"), 4)
)


def Q_builder(path: str) -> Callable[[np.ndarray], np.ndarray]:
    model = DQN("CnnPolicy", make_env(), buffer_size=0)
    model = model.load(path, buffer_size=0)

    def f(observation: np.ndarray) -> np.ndarray:
        obs = torch.tensor(observation, device=model.device)
        batched = len(obs.shape) == 4 and obs.shape[0] > 1
        if not batched and len(obs.shape) < 4:
            obs.unsqueeze_(0)
        obs.swapdims_(1, 3).swapdims_(2, 3)

        with torch.no_grad():
            q_values = model.q_net(obs).cpu().numpy()
        if batched:
            return q_values
        else:
            return q_values[0]

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


predicates = []


MY_PADDLE_X = 73
ENEMY_PADDLE_X = 10
PADDLE_HEIGHT = 7
BALL_SIZE = 1
# The lower the better, integer representing a pixel step
PREDICATE_STEP = 10


def get_paddle_y(paddle: int, img: np.ndarray) -> int:
    return np.argmax(img[14:76, paddle] != 87)


def get_ball_pos(img: np.ndarray, index: int = 3) -> Tuple[int, int]:
    tmp = img[14:76, ENEMY_PADDLE_X + 1 : MY_PADDLE_X, index] != 87
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


for i in range(ENEMY_PADDLE_X + 1, MY_PADDLE_X, PREDICATE_STEP):
    predicates.append(
        Predicate(f"ball x <= {i}", make_pred(i, lambda obs: get_ball_pos(obs)[0]))
    )

for i in range(0, 62, PREDICATE_STEP):
    predicates.append(
        Predicate(
            f"my paddle y <= {i}",
            make_pred(i, lambda obs: get_paddle_y(MY_PADDLE_X, obs)),
        )
    )
    predicates.append(
        Predicate(
            f"enemy paddle y <= {i}",
            make_pred(i, lambda obs: get_paddle_y(ENEMY_PADDLE_X, obs)),
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

predicates.append(
    Predicate(
        f"my paddle too low to hit ball",
        lambda obs: get_paddle_y(MY_PADDLE_X, ready(obs))
        > get_ball_pos(ready(obs))[1] + BALL_SIZE,
    )
)
predicates.append(
    Predicate(
        f"my paddle too high to hit ball",
        lambda obs: get_paddle_y(MY_PADDLE_X, ready(obs)) + PADDLE_HEIGHT
        < get_ball_pos(ready(obs))[1],
    )
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
#         plt.imshow(obs[14:76, ENEMY_PADDLE_X:MY_PADDLE_X + 1, i])
#     plt.show()
# print("my paddle:", get_paddle_y(MY_PADDLE_X, obs))
# print("enemy paddle:", get_paddle_y(ENEMY_PADDLE_X, obs))
# print("ball pos:", get_ball(obs))
# py =
# print(py)
