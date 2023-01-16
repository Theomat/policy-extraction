from typing import Any, Callable, List, Optional, TypeVar

from stable_baselines3.common.vec_env.dummy_vec_env import DummyVecEnv
from stable_baselines3.common.vec_env.vec_video_recorder import VecVideoRecorder


import numpy as np

S = TypeVar("S")
U = TypeVar("U")


def vec_eval_policy(
    policy: Callable[[np.ndarray], int],
    episodes: int,
    env_fn: Callable,
    nenvs: int,
    nactions: int,
    seed: Optional[int] = None,
) -> tuple[float, float]:
    def our_step(
        rew: float,
        ep: int,
        st: np.ndarray,
        q: np.ndarray,
        r: float,
        stp1: np.ndarray,
        done: bool,
    ) -> float:
        return rew + r

    total_rewards = vec_interact(
        policy_to_q_function(policy, nactions, nenvs),
        episodes,
        env_fn,
        nenvs,
        our_step,
        0.0,
        seed=seed,
    )
    return np.mean(total_rewards), 2 * np.std(total_rewards)


def policy_to_q_function(
    policy: Callable[[np.ndarray], int], nactions: int, nenvs: int
) -> Callable[[np.ndarray], np.ndarray]:
    def super_policy(states: np.ndarray) -> np.ndarray:
        actions = [policy(state) for state in states]
        return np.array(
            [
                [1 if actions[j] == i else 0 for i in range(nactions)]
                for j in range(nenvs)
            ]
        )

    return super_policy


def vec_interact(
    Q: Callable[[Any], np.ndarray],
    episodes: int,
    env_creator: Callable,
    nenv: int,
    step: Callable[[U, int, S, np.ndarray, float, S, bool], U],
    u0: U,
    seed: Optional[int] = None,
) -> List[U]:
    out = []
    venv = DummyVecEnv([env_creator for _ in range(nenv)])
    venv.seed(seed)
    current_episodes = [u0 for _ in range(nenv)]
    num_episodes = [i for i in range(nenv)]
    mask = [True for _ in range(nenv)]
    episodes_done = 0
    obs = venv.reset()
    while episodes_done < episodes:
        q_values = Q(obs)
        actions = np.argmax(q_values, axis=1)
        nobs, reward, done, info = venv.step(actions)
        for i in range(nenv):
            if not mask[i]:
                continue
            if done[i]:
                cobs = info[i]["terminal_observation"]
                val = step(
                    current_episodes[i],
                    num_episodes[i],
                    cobs,
                    q_values[i],
                    reward[i],
                    cobs,
                    True,
                )
                current_episodes[i] = u0
                out.append(val)
                num_episodes[i] = episodes_done + nenv
                episodes_done += 1
                mask[i] = num_episodes[i] < episodes
            else:
                current_episodes[i] = step(
                    current_episodes[i],
                    num_episodes[i],
                    obs[i],
                    q_values[i],
                    reward[i],
                    nobs[i],
                    done[i],
                )
        obs = nobs
    return out


def record_video(
    Q: Callable[[Any], np.ndarray],
    env_creator: Callable,
    nenv: int,
    video_folder: str,
    name_prefix: str = "agent",
    video_length: int = 100,
    seed: Optional[int] = None,
):
    venv = DummyVecEnv([env_creator for _ in range(nenv)])
    venv.seed(seed)
    venv = VecVideoRecorder(
        venv,
        video_folder,
        record_video_trigger=lambda x: x == 0,
        video_length=video_length,
        name_prefix=name_prefix,
    )

    obs = venv.reset()
    for _ in range(video_length + 1):
        q_values = Q(obs)
        actions = np.argmax(q_values, axis=1)
        obs, _, _, _ = venv.step(actions)
    # Save the video
    venv.close()
