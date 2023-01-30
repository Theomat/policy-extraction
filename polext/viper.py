from dataclasses import dataclass
from typing import Callable, Generator, List, Optional, Tuple, TypeVar, Union

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


@dataclass
class ViperAlgo:
    name: str
    method: Callable


__TREE_BUILDING_ALGOS__ = {}


def register(algorithm: ViperAlgo) -> None:
    __TREE_BUILDING_ALGOS__[algorithm.name.lower().strip()] = algorithm


def list_registered_algorithms() -> List[str]:
    return list(__TREE_BUILDING_ALGOS__.keys())


def __viper_wrapper__(name: str, build_tree: Callable) -> Callable:
    def f(
        space: PredicateSpace[S],
        dataset: List[Tuple[Tuple[bool, ...], np.ndarray, float]],
        max_depth: int,
        env_fn: Callable,
        nenvs: int,
        seed: Optional[int],
    ) -> DecisionTree[S]:
        new_space = PredicateSpace(space.predicates)
        qtable = QValuesLearner()
        for s, q, _ in dataset:
            ps = new_space.get_representative(s)
            qtable.add_one_visit(ps, q)
        for tree, _ in build_tree(
            new_space,
            qtable,
            max_depth,
            name,
            None,
            env_fn,
            nenvs,
            seed=seed,
            episodes=0,
        ):
            return tree

    return f


def __resample__(
    dataset: List[Tuple[Tuple[bool, ...], np.ndarray, float]], rng: np.random.Generator
) -> List[Tuple[Tuple[bool, ...], np.ndarray, float]]:
    n = len(dataset)
    probs = np.array([d for _, __, d in dataset])
    probs /= np.sum(probs)
    return rng.choice(np.array(dataset, dtype=object), size=n, p=probs, replace=True)  # type: ignore


def __best_policy__(
    policies: List[DecisionTree[S]],
    env_fn: Callable,
    nenvs: int,
    nactions: int,
    episodes: int,
    seed: Optional[int] = None,
) -> Tuple[DecisionTree[S], Tuple[float, float]]:
    best_policy = policies[0]
    best_score = -1e99
    values = (-1, -1)
    for policy in policies:
        score, std = vec_eval_policy(policy, episodes, env_fn, nenvs, nactions, seed)  # type: ignore
        if score > best_score:
            best_policy = policy
            best_score = score
            values = (score, std)
    return best_policy, values


def viper(
    space: PredicateSpace[S],
    qtable: QValuesLearner,
    max_depth: int,
    method: str,
    Qfun: Callable[[S], np.ndarray],
    env_fn: Callable,
    nenvs: int,
    iterations: int,
    samples: int,
    seed: Optional[int] = None,
) -> Generator[Tuple[DecisionTree[S], Tuple[float, float]], None, None]:
    dataset = []
    policy: Union[Callable[[S], np.ndarray], DecisionTree[S]] = Qfun
    policies = []

    basic_tree = __TREE_BUILDING_ALGOS__[method.lower().strip()].method
    rng = np.random.default_rng(seed)

    def my_step(
        u0,
        episode_no: int,
        state: S,
        Qvals: np.ndarray,
        reward: float,
        next_state: S,
        done: bool,
    ):
        if not done:
            true_Qvals = Qfun(state)
            dataset.append((state, true_Qvals, np.max(true_Qvals) - np.min(true_Qvals)))
        return u0

    for i in range(iterations):
        interact_policy = policy if i == 0 else policy_to_q_function(policy, qtable.nactions, nenvs)  # type: ignore
        vec_interact(
            interact_policy,  # type: ignore
            samples,
            env_fn,
            nenvs,
            my_step,
            0,
            seed,
        )
        dataset_prime = __resample__(dataset, rng)
        policy = basic_tree(space, dataset_prime, max_depth, env_fn, nenvs, seed)
        policies.append(policy)

    yield __best_policy__(policies, env_fn, nenvs, qtable.nactions, samples, seed)
