from typing import Any, Callable, Optional, Tuple, TypeVar
import copy

import numpy as np

from polext.decision_tree import DecisionTree
from polext.finite.tree_builder import (
    build_tree as fbuild_tree,
    build_forest as fbuild_forest,
)
from polext.forest import Forest, majority_vote
from polext.predicate_space import PredicateSpace
from polext.interaction_helper import (
    policy_to_q_function,
    vec_interact,
    vec_eval_policy,
)


S = TypeVar("S")
V = TypeVar("V")


def __iterate__(
    builder: Callable[[PredicateSpace[S], int, str], Tuple[V, Any]],
    space: PredicateSpace[S],
    max_depth: int,
    method: str,
    Qfun: Callable[[S], np.ndarray],
    env_fn: Callable,
    nenvs: int,
    iterations: int = 0,
    episodes: int = 0,
    seed: Optional[int] = None,
    **kwargs,
) -> Tuple[V, Tuple[float, float]]:
    tree, _ = builder(space, max_depth, method, seed=seed, **kwargs)
    if iterations <= 1:
        return tree, vec_eval_policy(
            tree, episodes, env_fn, nenvs, space.nactions, seed=seed
        )
    new_space = copy.deepcopy(space)
    new_space.reset_count()

    replay_buffer = []
    episodes_length = [0 for _ in range(episodes)]

    def our_step(
        rew: float, ep: int, st: S, Qval: np.ndarray, r: float, stp1: S, done: bool
    ) -> float:
        episodes_length[ep] += 1
        old_shape = st.shape
        qq = Qfun(st)
        if len(qq.shape) == 2:
            print("iterations:", iterations)
            print("st shape:", old_shape)
            print("qq shape:", qq.shape)
        new_space.visit_state(st, qq)
        action = np.argmax(Qval)
        replay_buffer.append((st, action, r, stp1, done))
        return rew + r

    total_rewards = vec_interact(
        policy_to_q_function(tree, space.nactions, min(nenvs, episodes)),
        episodes,
        env_fn,
        min(nenvs, episodes),
        our_step,
        0.0,
        seed=seed,
    )
    mu, std = np.mean(total_rewards), 2 * np.std(total_rewards)

    mean_length = np.mean(episodes_length)
    gamma = np.float_power(0.01, 1.0 / mean_length)

    for st, action, r, stp1, done in replay_buffer[::-1]:
        alpha = 1.0 / new_space.state_visits(st)
        new_space.learn_qvalues(st, action, r, stp1, done, alpha, gamma)

    new_space.mix_learnt(0.5, 0.5)

    def next_Q(state: S) -> np.ndarray:
        value = space.state_Q(state)
        return value if value is not None else Qfun(state)

    next_tree, (nmu, nstd) = __iterate__(
        builder,
        new_space,
        max_depth,
        method,
        Qfun=next_Q,
        iterations=iterations - 1,
        env_fn=env_fn,
        nenvs=nenvs,
        seed=seed,
        episodes=episodes,
        **kwargs,
    )

    if nmu > mu:
        return next_tree, (nmu, nstd)
    return tree, (mu, std)


def build_tree(
    space: PredicateSpace[S],
    max_depth: int,
    method: str,
    Qfun: Callable[[S], np.ndarray],
    env_fn: Callable,
    nenvs: int,
    iterations: int = 0,
    episodes: int = 0,
    seed: Optional[int] = None,
    **kwargs,
) -> Tuple[DecisionTree[S], Tuple[float, float]]:
    return __iterate__(
        fbuild_tree,
        space,
        max_depth,
        method,
        Qfun,
        env_fn,
        nenvs,
        iterations,
        episodes,
        seed,
        **kwargs,
    )


def build_forest(
    space: PredicateSpace[S],
    max_depth: int,
    method: str,
    trees: int,
    Qfun: Callable[[S], np.ndarray],
    env_fn: Callable,
    nenvs: int,
    iterations: int = 0,
    episodes: int = 0,
    seed: Optional[int] = None,
    **kwargs,
) -> Tuple[Forest[S], Tuple[float, float]]:
    return __iterate__(
        fbuild_forest,
        space,
        max_depth,
        method,
        Qfun,
        env_fn,
        nenvs,
        iterations,
        episodes,
        seed,
        trees=trees,
        **kwargs,
    )
