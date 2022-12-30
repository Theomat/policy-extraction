from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple, TypeVar

import numpy as np

from polext.decision_tree import DecisionTree
from polext.predicate_space import PredicateSpace
from polext.q_values_learner import QValuesLearner
from polext.interaction_helper import (
    policy_to_q_function,
    vec_interact,
    vec_eval_policy,
)

S = TypeVar("S")


def tree_loss(
    tree: DecisionTree[S], space: PredicateSpace[S], Qtable: QValuesLearner
) -> float:
    return sum(np.max(Qtable[s]) - Qtable[s][tree(s)] for s in space.seen)


@dataclass
class TreeBuildingAlgo:
    name: str
    method: Callable


__TREE_BUILDING_ALGOS__ = {}


def register(algorithm: TreeBuildingAlgo) -> None:
    __TREE_BUILDING_ALGOS__[algorithm.name.lower().strip()] = algorithm


def list_registered_algorithms() -> List[str]:
    return list(__TREE_BUILDING_ALGOS__.keys())


def __iterate__(
    builder: Callable[
        [PredicateSpace[S], QValuesLearner, int, int], Callable[[S], int]
    ],
    space: PredicateSpace[S],
    qtable: QValuesLearner,
    max_depth: int,
    method: str,
    Qfun: Callable[[S], np.ndarray],
    env_fn: Callable,
    nenvs: int,
    iterations: int = 0,
    episodes: int = 0,
    seed: Optional[int] = None,
    **kwargs,
) -> List[Tuple[Callable[[S], int], Tuple[float, float]]]:
    tree = builder(space, qtable, max_depth, seed=seed, **kwargs)
    if iterations <= 1:
        return [
            tree,
            vec_eval_policy(tree, episodes, env_fn, nenvs, qtable.nactions, seed=seed),
        ]
    next_qtable = QValuesLearner()

    replay_buffer = []
    episodes_length = [0 for _ in range(episodes)]

    def our_step(
        rew: float, ep: int, st: S, Qval: np.ndarray, r: float, stp1: S, done: bool
    ) -> float:
        episodes_length[ep] += 1
        qq = Qfun(st)
        ps = space.get_representative(st)
        next_qtable.add_one_visit(ps, qq)
        action = np.argmax(Qval)
        replay_buffer.append((ps, action, r, space.get_representative(stp1), done))
        return rew + r

    total_rewards = vec_interact(
        policy_to_q_function(tree, qtable.nactions, min(nenvs, episodes)),
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

    next_qtable.reset_Q()

    for st, action, r, stp1, done in replay_buffer[::-1]:
        alpha = 1.0 / next_qtable.state_visits(st)
        next_qtable.learn_qvalues(st, action, r, stp1, done, alpha, gamma)

    next_qtable.mix_with(qtable, 0.5)

    def next_Q(state: S) -> np.ndarray:
        value = next_qtable.state_normalised_Q(space.get_representative(state))
        return value if value is not None else Qfun(state)

    next_results = __iterate__(
        builder,
        space,
        next_qtable,
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
    return [tree, (mu, std)] + next_results


def build_tree(
    space: PredicateSpace[S],
    qtable: QValuesLearner,
    max_depth: int,
    method: str,
    Qfun: Callable[[S], np.ndarray],
    env_fn: Callable,
    nenvs: int,
    iterations: int = 0,
    episodes: int = 0,
    seed: Optional[int] = None,
    **kwargs,
) -> List[Tuple[DecisionTree[S], Tuple[float, float]]]:
    return __iterate__(
        __TREE_BUILDING_ALGOS__[method.lower().strip()].method,
        space,
        qtable,
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
