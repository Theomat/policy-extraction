from typing import Callable, Dict, Iterable, List, Tuple, TypeVar

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


def forest_loss(
    forest: Forest[S], Qtable: Dict[S, List[float]], states: Iterable[S]
) -> float:
    lost_reward = sum(
        max(Qtable[s]) - Qtable[s][majority_vote(forest(s))] for s in states
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
    states: List[S],
    Q: Callable[[S], List[float]],
    predicates: List[Predicate[S]],
    max_depth: int,
    method: str,
    **kwargs
) -> Tuple[DecisionTree[S], float]:
    space = PredicateSpace(predicates)
    for s in states:
        space.visit_state(s, Q(s))
    tree = _METHODS_[method.lower().strip()](space, max_depth, **kwargs).simplified()
    return tree, tree_loss(tree, space)


def build_forest(
    states: List[S],
    Q: Callable[[S], List[float]],
    predicates: List[Predicate[S]],
    max_depth: int,
    method: str,
    trees: int,
    seed: int,
    **kwargs
) -> Tuple[DecisionTree[S], float]:
    Qtable = {s: Q(s) for s in states}
    predicates_table = {
        predicate: {s for s in states if predicate(s)} for predicate in predicates
    }
    nactions = len(Qtable[states[0]])
    sample_size = int(np.floor(np.sqrt(len(states))))

    rng = np.random.default_rng(seed)

    def gen_tree() -> DecisionTree[S]:
        sub_states = {
            tuple(s) for s in rng.choice(states, size=sample_size, replace=False)
        }
        return _METHODS_[method.lower().strip()](
            sub_states,
            {s: q for s, q in Qtable.items() if s in sub_states},
            {p: sub_states.intersection(s) for p, s in predicates_table.items()},
            nactions,
            max_depth,
            seed=seed,
            **kwargs
        ).simplified()

    trees = [gen_tree() for _ in range(trees)]
    forest = Forest(trees)
    return forest, forest_loss(forest, Qtable, states)
