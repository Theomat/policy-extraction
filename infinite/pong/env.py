from typing import Callable, List, Tuple
import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env.base_vec_env import (
    VecEnvWrapper,
    VecEnv,
    VecEnvStepReturn,
    VecEnvObs,
)
from stable_baselines3.common.env_util import make_atari_env
from gym.spaces.box import Box
import gym

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


MAX_Y = 63.0
MY_PADDLE_X = 73
ENEMY_PADDLE_X = 10
X_LENGTH = MY_PADDLE_X - ENEMY_PADDLE_X


def get_paddle_y(paddle: int, img: np.ndarray) -> int:
    return np.argmax(img[:, paddle] != 87)  # type: ignore


def get_ball_pos(img: np.ndarray) -> Tuple[int, int]:
    tmp = img[:, ENEMY_PADDLE_X + 1 : MY_PADDLE_X] != 87
    y = np.max(np.argmax(tmp, axis=0, keepdims=True))
    return np.argmax(tmp[y]) + ENEMY_PADDLE_X + 1, y  # type: ignore


def __extract__(obs: np.ndarray) -> List:
    if len(obs.shape) == 4:
        return [__extract__(v) for v in obs]
    obs = obs[14:76]
    y_paddle = get_paddle_y(MY_PADDLE_X, obs)
    x, y = get_ball_pos(obs)
    return [y_paddle / MAX_Y, (x - (ENEMY_PADDLE_X + 1)) / X_LENGTH, y / MAX_Y]


def __obs__(pobs: List, history: List[np.ndarray], add: bool = True) -> np.ndarray:
    if isinstance(pobs[0], List):
        out = np.asarray(
            [
                __obs__(pobs[i], [x[i] for x in history], add=False)
                for i in range(len(pobs))
            ]
        )
    else:
        out = np.zeros((8), dtype=float)
        # ball x, y
        out[0] = pobs[1]
        out[1] = pobs[2]
        # paddle y
        out[4] = pobs[0]
        if len(history) > 0:
            # paddle vy
            out[5] = out[4] - history[-1][4]
            # ball vx, vy
            out[2] = out[0] - history[-1][0]
            out[3] = out[1] - history[-1][1]
            if len(history) > 1:
                # paddle ay
                out[6] = out[5] - history[-2][5]
                if len(history) > 2:
                    # paddle jy
                    out[7] = out[6] - history[-3][6]

    if add:
        history.append(out)
        if len(history) > 3:
            history.pop(0)
    return out


class RLZooObservationWrapper(gym.ObservationWrapper):
    """
    Used only by RL ZOO to train pong
    """

    def __init__(self, env: gym.Env):
        gym.ObservationWrapper.__init__(self, env)
        self.observation_space = Box(
            np.array([0, 0, -1, -1, 0, -1, -2, -4.0]),
            np.array([1, 1, 1, 1, 1, 1, 2, 4.0]),
            dtype=float,
        )
        self.history = []

    def observation(self, obs: np.ndarray) -> np.ndarray:
        pobs = __extract__(obs)
        nobs = __obs__(pobs, self.history)
        return nobs

    def reset(self) -> np.ndarray:
        self.history = []
        return super().reset()


class VecPongObservationWrapper(VecEnvWrapper):
    def __init__(self, env: VecEnv) -> None:
        super().__init__(env)
        self.observation_space = Box(
            np.array([0, 0, -1, -1, 0, -1, -2, -4.0]),
            np.array([1, 1, 1, 1, 1, 1, 2, 4.0]),
            dtype=float,
        )
        self.history = []

    def step_async(self, actions: np.ndarray) -> None:
        return self.venv.step_async(actions)

    def step_wait(self) -> VecEnvStepReturn:
        obs, r, done, info = self.venv.step_wait()
        pobs = __extract__(obs)
        nobs = __obs__(pobs, self.history)
        return nobs, r, done, info

    def reset(self) -> VecEnvObs:
        obs = self.venv.reset()
        pobs = __extract__(obs)
        self.history = []
        nobs = __obs__(pobs, self.history)
        return nobs


make_env = lambda: VecActionWrapper(
    VecPongObservationWrapper(make_atari_env("PongNoFrameskip-v4"))
)


def Q_builder(path: str) -> Callable[[np.ndarray], np.ndarray]:
    model = DQN("MlpPolicy", make_env(), buffer_size=0)
    model = model.load(path, buffer_size=0)

    def f(observation: np.ndarray) -> np.ndarray:
        obs = torch.tensor(observation, device=model.device)
        batched = len(obs.shape) == 2 and obs.shape[0] > 1
        if not batched:
            obs.unsqueeze_(0)
        with torch.no_grad():
            q_values = model.q_net(obs).cpu().numpy()
        if batched:
            return q_values
        else:
            return q_values[0]

    return f


predicates = []


PADDLE_HEIGHT = 7.0 / MAX_Y
BALL_SIZE = 1.0 / MAX_Y

bins = 10

states_arrays = [
    ("ball x", np.linspace(0, +1.0, num=bins, endpoint=False)),
    ("ball y", np.linspace(0, +1.0, num=bins, endpoint=False)),
    ("ball vx", np.linspace(-0.1, +1.0, num=bins, endpoint=False)),
    ("ball vy", np.linspace(-1.0, +1.0, num=bins, endpoint=False)),
    ("paddle y", np.linspace(0, +1, num=bins, endpoint=False)),
    ("paddle vy", np.linspace(-1, +1, num=bins, endpoint=False)),
    ("paddle ay", np.linspace(-1, +1, num=bins, endpoint=False)),
    ("paddle jy", np.linspace(-1, +1, num=bins, endpoint=False)),
]


def __ready__(f):
    def g(s):
        if len(s.shape) == 2:
            return f(s[0])
        return f(s)

    return g


def future_paddle_y(obs: np.ndarray, time: int):
    y, vy, ay, jy = obs[-4], obs[-3], obs[-2], obs[-1]
    return y + vy * time + ay / 2 * (time**2) + jy / 6 * (time**3)


def when_hit(obs: np.ndarray) -> int:
    x = obs[0]
    vx = obs[2]
    time = 0
    while x < 1.0:
        x += vx
        time += 1
        if x < 0:
            vx *= -1
            x = -x
    return x


def future_ball_y(obs: np.ndarray, time: int):
    y = obs[0]
    vy = obs[2]
    y += vy * time
    y -= int(y)
    return y


def pred(i: int, val: float):
    def f(s) -> bool:
        return s[i] >= val

    return f


predicates = []
for i, (name, array) in enumerate(states_arrays):
    for el in array:
        predicates.append(Predicate(f"{name} >= {el:.2e}", __ready__(pred(i, el))))

predicates.append(
    Predicate(
        f"paddle too low to hit ball",
        __ready__(lambda obs: obs[1] - obs[4] > PADDLE_HEIGHT),
    )
)
predicates.append(
    Predicate(
        f"paddle too high to hit ball",
        __ready__(lambda obs: obs[1] - obs[4] + BALL_SIZE < 0),
    )
)

predicates.append(
    Predicate(
        f"paddle when hit too low to hit ball",
        __ready__(
            lambda obs: future_ball_y(obs, when_hit(obs))
            - future_paddle_y(obs, when_hit(obs))
            > PADDLE_HEIGHT
        ),
    )
)
predicates.append(
    Predicate(
        f"paddle when hit too high to hit ball",
        __ready__(
            lambda obs: future_ball_y(obs, when_hit(obs))
            - future_paddle_y(obs, when_hit(obs))
            + BALL_SIZE
            < 0
        ),
    )
)

if __name__ == "__main__":
    from matplotlib import pyplot as plt

    env = make_env()
    obs = env.reset()
    for i in range(400):
        obs = env.step([0])[0]
        print(obs)
    # print(obs.shape)
#     for predicate in predicates:
#         print(predicate.name, predicate(obs))
#     plt.figure()
#     for i in range(4):
#         plt.subplot(1, 4, 1 + i)
#         plt.imshow(obs[14:76, ENEMY_PADDLE_X:MY_PADDLE_X + 1, i])
#     plt.show()
# print("my paddle:", get_paddle_y(MY_PADDLE_X, obs))
# print("enemy paddle:", get_paddle_y(ENEMY_PADDLE_X, obs))
# print("ball pos:", get_ball(obs))
