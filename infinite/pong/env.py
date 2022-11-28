from typing import Callable, List, Tuple
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
        with torch.no_grad():
            q_values = model.q_net(observation)[0]
        return [x.item() for x in q_values]

    return f


predicates = [
    Predicate(
        "my paddle y < ball y",
        lambda obs: get_paddle_y(MY_PADDLE, obs) < get_ball(obs)[1],
    ),
]


MY_PADDLE = 73
ENEMY_PADDLE = 10
PADDLE_HEIGHT = 7


def get_paddle_y(paddle: int, img: np.ndarray) -> int:
    return np.argmax(img[14:76, paddle] != 87)


def get_ball(img: np.ndarray) -> Tuple[int, int]:
    tmp = img[14:76, ENEMY_PADDLE + 1 : MY_PADDLE] != 87
    x = np.max(np.argmax(tmp, axis=0, keepdims=True))
    return x, np.argmax(tmp[x])


def make_pred(val: int, getter: Callable):
    def f(obs):
        return val <= getter(obs)

    return f


for i in range(ENEMY_PADDLE + 1, MY_PADDLE, 10):
    predicates.append(
        Predicate(f"ball x < {i}", make_pred(i, lambda obs: get_ball(obs)[0]))
    )
for i in range(0, 62, 10):
    predicates.append(
        Predicate(
            f"my paddle y < {i}", make_pred(i, lambda obs: get_paddle_y(MY_PADDLE, obs))
        )
    )
    predicates.append(
        Predicate(
            f"enemy paddle y < {i}",
            make_pred(i, lambda obs: get_paddle_y(ENEMY_PADDLE, obs)),
        )
    )
    predicates.append(
        Predicate(f"ball y < {i}", make_pred(i, lambda obs: get_ball(obs)[1]))
    )

# if __name__ == "__main__":
#     env = make_env()
#     obs = env.reset()
#     for i in range(400):
#         obs = env.step(0)[0]
#     new_obs = obs[14:76] != 87
#     print("my paddle:", get_paddle_y(MY_PADDLE, obs))
#     print("enemy paddle:", get_paddle_y(ENEMY_PADDLE, obs))
#     print("ball pos:", get_ball(obs))
#     # py =
#     # print(py)
