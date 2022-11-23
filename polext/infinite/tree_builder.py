from typing import Any, Callable, List, Optional, Tuple, TypeVar
import copy

import numpy as np

from polext.decision_tree import DecisionTree
from polext.finite.tree_builder import (
    build_tree as fbuild_tree,
    build_forest as fbuild_forest,
)
from polext.forest import Forest, majority_vote
from polext.predicate_space import PredicateSpace
from polext.interaction_helper import interact, eval_policy


S = TypeVar("S")


def build_tree(
    space: PredicateSpace[S],
    max_depth: int,
    method: str,
    Qfun: Callable[[S], List[float]],
    env: Any,
    iterations: int = 0,
    episodes: int = 0,
    **kwargs
) -> Tuple[DecisionTree[S], Tuple[float, float]]:
    tree, _ = fbuild_tree(space, max_depth, method, **kwargs)
    if iterations <= 1:
        return tree, eval_policy(tree, episodes, env)
    new_space = copy.deepcopy(space)
    new_space.reset_count()

    def our_step(
        rew: List[float], ep: int, st: S, r: float, stp1: S, done: bool
    ) -> List[float]:
        rew[-1] += r
        new_space.visit_state(st, Qfun(st))
        if done:
            rew.append(0)
        return rew

    total_rewards = interact(tree, episodes, env, our_step, [0.0])
    mu, std = np.mean(total_rewards), 2 * np.std(total_rewards)

    next_tree, (nmu, nstd) = build_tree(
        new_space,
        max_depth,
        method,
        Qfun=Qfun,
        iterations=iterations - 1,
        env=env,
        episodes=episodes,
        **kwargs
    )

    if nmu > mu:
        return next_tree, (nmu, nstd)
    return tree, (mu, std)


def build_forest(
    space: PredicateSpace[S],
    max_depth: int,
    method: str,
    trees: int,
    seed: int,
    iterations: int = 0,
    env: Optional[Any] = None,
    **kwargs
) -> Tuple[Forest[S], float]:
    gen = space.random_splits(seed)
    our_trees = [
        build_tree(
            next(gen),
            max_depth=max_depth,
            method=method,
            seed=seed + i,
            iterations=iterations,
            env=env,
            **kwargs
        )[0]
        for i in range(trees)
    ]
    forest = Forest(our_trees)
    return forest, forest_loss(forest, space)
