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
        return self.venv.step([action])

    def step_wait(self) -> VecEnvStepReturn:
        return self.venv.step_wait()

    def reset(self) -> VecEnvObs:
        return self.venv.reset()


make_env = lambda: VecActionWrapper(
    VecFrameStack(make_atari_env("PongNoFrameskip-v4"), 4)
)


def Q_builder(path: str) -> Callable[[np.ndarray], List[float]]:
    model = DQN("CnnPolicy", make_env(), buffer_size=0)
    model = model.load(path)

    def f(observation: np.ndarray) -> List[float]:
        observation = (
            torch.tensor(observation, device=model.device)[0].swapdims_(0, 2)
        ).unsqueeze_(0)
        with torch.no_grad():
            q_values = model.q_net(observation)[0]
        return [x.item() for x in q_values]

    return f


predicates = [
    Predicate(
        "my paddle y < ball y",
        lambda obs: get_paddle_y(MY_PADDLE, obs[0]) < get_ball_pos(obs[0])[1],
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
        return val <= getter(obs[0])

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
