from dataclasses import dataclass
from typing import Callable, Generator, List, Optional, Tuple, TypeVar

import numpy as np

from polext.decision_tree import DecisionTree
from polext.forest import Forest
from polext.predicate_space import PredicateSpace
from polext.q_values_learner import QValuesLearner
from polext.interaction_helper import (
    policy_to_q_function,
    vec_interact,
    vec_eval_policy,
)

S = TypeVar("S")


def regret(
    tree: DecisionTree[S], space: PredicateSpace[S], Qtable: QValuesLearner
) -> float:
    regret = 0
    for s in space.seen:
        Qvals = Qtable[s]
        if Qvals is None:
            continue
        regret += np.max(Qvals) - Qvals[tree.eval_pred_space(s, space)]
    return regret


def tree_loss(
    tree: DecisionTree[S], space: PredicateSpace[S], Qtable: QValuesLearner
) -> float:
    loss = 0
    for s in space.seen:
        loss += -Qtable[s, tree(s, space)]
    return loss


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
) -> Generator[Tuple[Callable[[S], int], Tuple[float, float]], None, None]:
    tree = builder(space, qtable, max_depth, seed=seed, **kwargs)  # type: ignore
    tree.seed(seed)
    actual_nenvs = min(nenvs, episodes)
    if iterations <= 1:

        yield (
            tree,
            vec_eval_policy(
                tree,
                episodes,
                env_fn,
                actual_nenvs,
                qtable.nactions if not isinstance(qtable, List) else qtable[0].nactions,
                seed=seed,
            ),
        )
        return

    next_qtable = QValuesLearner()
    new_space = PredicateSpace(space.predicates)

    replay_buffer = []
    episodes_length = [0 for _ in range(episodes)]

    def our_step(
        rew: float, ep: int, st: S, Qval: np.ndarray, r: float, stp1: S, done: bool
    ) -> float:
        episodes_length[ep] += 1
        ps = new_space.get_representative(st)
        oracle_q = Qfun(st)
        next_qtable.add_one_visit(ps, oracle_q)
        action = np.argmax(Qval)
        replay_buffer.append(
            (ps, action, r, new_space.get_representative(stp1, False), done)
        )
        return rew + r

    total_rewards = vec_interact(
        policy_to_q_function(tree, qtable.nactions, actual_nenvs),
        episodes,
        env_fn,
        actual_nenvs,
        our_step,
        0.0,
        seed=seed,
    )
    mu, std = np.mean(total_rewards), 2 * np.std(total_rewards)
    yield (tree, (mu, std))  # type: ignore

    # Learning Q_Value in predicate space
    mean_length = np.mean(episodes_length)
    gamma = np.float_power(0.05, 1.0 / mean_length)

    # next_qtable.reset_Q()

    for st, action, r, stp1, done in replay_buffer[::-1]:
        alpha = 0.25 / next_qtable.state_visits(st)
        next_qtable.learn_qvalues(st, action, r, stp1, done, alpha, gamma)

    # Mix the two Qtables
    # next_qtable.mix_with(qtable, 0.5)

    def next_Q(state: S) -> np.ndarray:
        value = next_qtable[new_space.get_representative(state, False)]
        return value if value is not None else Qfun(state)

    for x in __iterate__(
        builder,
        new_space,
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
    ):
        yield x


def __forest__(
    builder: Callable[[PredicateSpace[S], QValuesLearner, int, int], DecisionTree[S]],
    ntrees: int,
) -> Callable[[PredicateSpace[S], QValuesLearner, int, int], Forest[S]]:
    def f(
        space: PredicateSpace[S],
        qtable: QValuesLearner,
        max_depth: int,
        seed: int,
        **kwargs,
    ):
        gen = space.random_splits(seed)
        return Forest(
            [
                builder(next(gen), qtable, max_depth, seed=seed, **kwargs)  # type: ignore
                for _ in range(ntrees)
            ]
        )

    return f


def build_tree(
    space: PredicateSpace[S],
    qtable: QValuesLearner,
    max_depth: int,
    method: str,
    Qfun: Callable[[S], np.ndarray],
    env_fn: Callable,
    nenvs: int,
    iterations: int = 1,
    episodes: int = 0,
    trees: int = 1,
    seed: Optional[int] = None,
    **kwargs,
) -> Generator[Tuple[DecisionTree[S], Tuple[float, float]], None, None]:
    basic_tree = __TREE_BUILDING_ALGOS__[method.lower().strip()].method
    return __iterate__(
        basic_tree if trees <= 1 else __forest__(basic_tree, trees),
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
    )  # type: ignore
