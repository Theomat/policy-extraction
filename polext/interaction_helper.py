from typing import Any, Callable, List, Optional, TypeVar

from stable_baselines3.common.vec_env.dummy_vec_env import DummyVecEnv


import numpy as np

S = TypeVar("S")
U = TypeVar("U")


def interact(
    policy: Callable[[S], int],
    episodes: int,
    env,
    step: Callable[[U, int, S, float, S, bool], U],
    u0: U,
) -> U:
    val = u0
    for ep in range(episodes):
        done = False
        state = env.reset()
        while not done:
            next_state, reward, done, _ = env.step(policy(state))
            val = step(val, ep, state, reward, next_state, done)
            state = next_state
    return val


def eval_policy(
    policy: Callable[[np.ndarray], int], episodes: int, env
) -> tuple[float, float]:
    def our_step(
        rew: List[float],
        ep: int,
        st: np.ndarray,
        r: float,
        stp1: np.ndarray,
        done: bool,
    ) -> List[float]:
        rew[-1] += r
        if done:
            rew.append(0)
        return rew

    total_rewards = interact(policy, episodes, env, our_step, [0.0])
    total_rewards.pop()
    return np.mean(total_rewards), 2 * np.std(total_rewards)


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

    def super_policy(states: np.ndarray) -> np.ndarray:
        actions = [policy(state) for state in states]
        return np.array(
            [
                [1 if actions[j] == i else 0 for i in range(nactions)]
                for j in range(nenvs)
            ]
        )

    total_rewards = vec_interact(
        super_policy, episodes, env_fn, nenvs, our_step, 0.0, seed=seed
    )
    return np.mean(total_rewards), 2 * np.std(total_rewards)


def eval_q(Q: Callable[[Any], List[float]], episodes: int, env) -> tuple[float, float]:
    def f(state) -> int:
        action = np.argmax(Q(state))
        if isinstance(action, np.ndarray):
            action = action[0]
        return action

    return eval_policy(f, episodes, env)


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
                    episodes_done,
                    cobs,
                    q_values[i],
                    reward[i],
                    cobs,
                    True,
                )
                current_episodes[i] = u0
                out.append(val)
                episodes_done += 1
                mask[i] = episodes_done < episodes
            else:
                current_episodes[i] = step(
                    current_episodes[i],
                    episodes_done,
                    obs[i],
                    q_values[i],
                    reward[i],
                    nobs[i],
                    done[i],
                )
        obs = nobs
    return out
