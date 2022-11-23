from typing import Callable, List, Tuple, TypeVar

import numpy as np

from polext.decision_tree import DecisionTree
from polext.finite.greedy_builder import (
    greedy_tree_builder,
    greedy_q_selection,
    greedy_opt_action_selection,
)
from polext.finite.max_probability_tree_builder import max_probability_tree_builder
from polext.finite.optimistic_tree_builder import optimistic_tree_builder
from polext.finite.simulated_annealing_tree_builder import (
    simulated_annealing_tree_builder,
)
from polext.forest import Forest, majority_vote
from polext.predicate import Predicate
from polext.predicate_space import PredicateSpace

S = TypeVar("S")


def tree_loss(tree: DecisionTree[S], space: PredicateSpace[S]) -> float:
    lost_reward = sum(
        max(space.Qtable[s]) - space.Qtable[s][tree(s)] for s in space.states
    )
    return lost_reward


def forest_loss(forest: Forest[S], space: PredicateSpace[S]) -> float:
    lost_reward = sum(
        max(space.Qtable[s]) - space.Qtable[s][majority_vote(forest(s))]
        for s in space.states
    )
    return lost_reward


_METHODS_ = {
    "greedy-q": greedy_tree_builder(greedy_q_selection, tree_loss),
    "greedy-nactions": greedy_tree_builder(greedy_opt_action_selection, tree_loss),
    "max-probability": max_probability_tree_builder(tree_loss),
    "optimistic": optimistic_tree_builder(tree_loss),
    "simulated-annealing": simulated_annealing_tree_builder(
        tree_loss, 0.5, 200, 100, 0.2, 100
    ),
}


def build_tree(
    space: PredicateSpace[S], max_depth: int, method: str, **kwargs
) -> Tuple[DecisionTree[S], float]:
    tree = _METHODS_[method.lower().strip()](space, max_depth, **kwargs).simplified()
    return tree, tree_loss(tree, space)


def easy_space(
    states: List[S],
    Q: Callable[[S], List[float]],
    predicates: List[Predicate[S]],
    use_representatives: bool = False,
) -> PredicateSpace[S]:
    space = PredicateSpace(predicates, use_representatives)
    for s in states:
        space.visit_state(s, Q(s))
    return space


def interactive_space(
    states: List[S],
    Q: Callable[[S], List[float]],
    predicates: List[Predicate[S]],
    env,
    episodes: int,
    use_representatives: bool = False,
) -> PredicateSpace[S]:
    space = PredicateSpace(predicates, use_representatives)
    for s in states:
        space.visit_state(s, Q(s))

    for _ in range(episodes):
        state = env.reset()
        done = False
        while not done:
            Qvalues = Q(state)
            space.visit_state(state, Qvalues)
            action = np.argmax(Qvalues)
            state, _, done, _ = env.step(action)

    return space


def build_forest(
    space: PredicateSpace[S],
    max_depth: int,
    method: str,
    trees: int,
    seed: int,
    **kwargs
) -> Tuple[Forest[S], float]:
    gen = space.random_splits(seed)
    our_trees = [
        build_tree(
            next(gen), max_depth=max_depth, method=method, seed=seed + i, **kwargs
        )[0]
        for i in range(trees)
    ]
    forest = Forest(our_trees)
    return forest, forest_loss(forest, space)
