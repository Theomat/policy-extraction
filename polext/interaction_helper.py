from typing import Any, Callable, List, TypeVar

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


def eval_q(Q: Callable[[Any], List[float]], episodes: int, env) -> tuple[float, float]:
    def f(state) -> int:
        action = np.argmax(Q(state))
        if isinstance(action, np.ndarray):
            action = action[0]
        return action

    return eval_policy(f, episodes, env)
