from typing import Dict, Optional, Set, TypeVar

import numpy as np

from polext.decision_tree import DecisionTree, Node
from polext.algos.greedy import greedy_finisher
from polext.predicate import Predicate
from polext.predicate_space import PredicateSpace
from polext.q_values_learner import QValuesLearner
from polext.tree_builder import register, TreeBuildingAlgo

S = TypeVar("S")


def max_probability_tree_builder(
    space: PredicateSpace[S], Qtable: QValuesLearner, max_depth: int, **kwargs
) -> DecisionTree[S]:
    Qmax = {s: np.argmax(Qtable[s]) for s in space}
    classes = {action: set() for action in range(Qtable.nactions)}
    for s in space:
        a = Qmax[s]
        classes[a].add(s)
    return __builder__(
        space,
        Qtable,
        max_depth,
        Qmax,
        classes,
    ).simplified()


register(TreeBuildingAlgo("max-probability", max_probability_tree_builder))


def __compute_score__(
    space: PredicateSpace[S],
    Qtable: QValuesLearner,
    candidate: Optional[Predicate[S]],
    depth_left: int,
    Qmax: Dict[S, int],
    classes: Dict[int, Set[S]],
) -> float:
    if len(space.seen) == 0:
        return 0
    score = 0
    pos = space.seen if candidate is None else space.predicates_set[candidate]
    neg = [] if candidate is None else list(space.predicate_set_complement(candidate))
    part_classes = {
        action: sum(Qtable.state_probability(s) for s in pos if s in sub)
        for action, sub in classes.items()
    }
    not_part_classes = {
        action: sum(Qtable.state_probability(s) for s in neg if s in sub)
        for action, sub in classes.items()
    }
    tpart = max(1e-99, sum(part_classes.values()))
    tnpart = max(1e-99, sum(not_part_classes.values()))
    for s in pos:
        a = Qmax[s]
        score += part_classes[a] / tpart * Qtable[s][a]
    for s in neg:
        a = Qmax[s]
        score += not_part_classes[a] / tnpart * Qtable[s][a]
    return score


def __builder__(
    space: PredicateSpace[S],
    Qtable: QValuesLearner,
    depth_left: int,
    Qmax: Dict[S, int],
    classes: Dict[int, Set[S]],
) -> DecisionTree[S]:
    if depth_left <= 2:
        tree = greedy_finisher(space, Qtable)
        assert tree
        return tree

    # Compute current score
    best_predicate = None
    best_score = __compute_score__(space, Qtable, None, depth_left, Qmax, classes)

    for candidate in space.unused_predicates():
        score = __compute_score__(space, Qtable, candidate, depth_left, Qmax, classes)
        if score > best_score:
            best_score = score
            best_predicate = candidate

    if best_predicate is None:
        tree = greedy_finisher(space, Qtable)
        assert tree
        return tree
    sub_states = space.predicates_set[best_predicate]
    left_space, right_space = space.split(best_predicate)
    left = __builder__(
        left_space,
        Qtable,
        depth_left - 1,
        Qmax,
        {action: sub.intersection(sub_states) for action, sub in classes.items()},
    )
    right = __builder__(
        right_space,
        Qtable,
        depth_left - 1,
        Qmax,
        {action: sub.difference(sub_states) for action, sub in classes.items()},
    )
    return Node(best_predicate, left, right)
