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
    **kwargs
) -> Tuple[V, Tuple[float, float]]:
    tree, _ = builder(space, max_depth, method, seed=seed, **kwargs)
    if iterations <= 1:
        return tree, vec_eval_policy(
            tree, episodes, env_fn, nenvs, space.nactions, seed
        )
    new_space = copy.deepcopy(space)
    new_space.reset_count()

    def our_step(
        rew: float, ep: int, st: S, Qval: np.ndarray, r: float, stp1: S, done: bool
    ) -> float:
        new_space.visit_state(st, Qfun(st))
        new_space.learn_qvalues(st, np.argmax(Qval), r, stp1, done)
        return rew + r

    total_rewards = vec_interact(
        policy_to_q_function(tree, space.nactions, nenvs),
        episodes,
        env_fn,
        nenvs,
        our_step,
        0.0,
    )
    mu, std = np.mean(total_rewards), 2 * np.std(total_rewards)
    new_space.mix_learnt(0.5, 0.5)
    next_tree, (nmu, nstd) = __iterate__(
        builder,
        new_space,
        max_depth,
        method,
        Qfun=Qfun,
        iterations=iterations - 1,
        env_fn=env_fn,
        nenvs=nenvs,
        seed=seed,
        episodes=episodes,
        **kwargs
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
    **kwargs
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
        **kwargs
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
    **kwargs
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
        **kwargs
    )
