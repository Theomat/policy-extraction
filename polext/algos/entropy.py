from typing import Dict, Set, TypeVar

import numpy as np

from polext.decision_tree import DecisionTree, Node
from polext.algos.greedy import greedy_finisher
from polext.predicate_space import PredicateSpace
from polext.q_values_learner import QValuesLearner
from polext.tree_builder import register, TreeBuildingAlgo

S = TypeVar("S")


def entropy_tree_builder(
    space: PredicateSpace[S], Qtable: QValuesLearner, max_depth: int, **kwargs
) -> DecisionTree[S]:
    classes = {action: set() for action in range(Qtable.nactions)}
    for s in space.seen:
        a = np.argmax(Qtable[s])
        classes[a].add(s)
    return __builder__(
        space,
        Qtable,
        max_depth,
        classes,
    ).simplified()


register(TreeBuildingAlgo("entropy", entropy_tree_builder))


def __compute_score__(
    space: PredicateSpace[S],
    Qtable: QValuesLearner,
    sub_states: Set[S],
    classes: Dict[int, Set[S]],
) -> float:
    if len(space.seen) == 0:
        return 0
    score = 0
    part_classes = {
        action: sum(Qtable.state_probability(x) for x in sub_states)
        / max(1, len(sub_states))
        for action, _ in classes.items()
    }
    not_part_classes = {
        action: sum(
            Qtable.state_probability(s) for s in space.seen if s not in sub_states
        )
        / max(1, (len(space.seen) - len(sub_states)))
        for action, _ in classes.items()
    }
    pos_entropy = 0
    neg_entropy = 0
    for a in part_classes.keys():
        if part_classes[a] > 0:
            pos_entropy += part_classes[a] * np.log2(part_classes[a])
        if not_part_classes[a] > 0:
            neg_entropy += not_part_classes[a] * np.log2(not_part_classes[a])
    score = (
        len(sub_states) / len(space.seen) * pos_entropy
        + (1 - len(sub_states) / len(space.seen)) * neg_entropy
    )
    return -score


def __builder__(
    space: PredicateSpace[S],
    Qtable: QValuesLearner,
    depth_left: int,
    classes: Dict[int, Set[S]],
) -> DecisionTree[S]:
    if depth_left <= 2:
        tree = greedy_finisher(space, Qtable)
        assert tree
        return tree

    # Compute current score
    best_predicate = None
    best_score = __compute_score__(space, Qtable, space.seen, classes)

    for candidate, sub_states in space.predicates_set.items():

        score = __compute_score__(space, Qtable, sub_states, classes)
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
        {action: sub.intersection(sub_states) for action, sub in classes.items()},
    )
    right = __builder__(
        right_space,
        Qtable,
        depth_left - 1,
        {action: sub.difference(sub_states) for action, sub in classes.items()},
    )
    return Node(best_predicate, left, right)
