from typing import Dict, Set, TypeVar

import numpy as np

from polext.decision_tree import DecisionTree, Node
from polext.algos.greedy import greedy_finisher
from polext.predicate_space import PredicateSpace
from polext.q_values_learner import QValuesLearner
from polext.tree_builder import register, TreeBuildingAlgo

S = TypeVar("S")


def optimistic_tree_builder(
    space: PredicateSpace[S], Qtable: QValuesLearner, max_depth: int, **kwargs
) -> DecisionTree[S]:
    Qmax = {s: np.argmax(Qtable[s]) for s in space.seen}
    classes = {action: set() for action in range(Qtable.nactions)}
    for s in space.seen:
        a = Qmax[s]
        classes[a].add(s)
    return __builder__(
        space,
        Qtable,
        max_depth,
        Qmax,
        classes,
    ).simplified()


register(TreeBuildingAlgo("optimistic", optimistic_tree_builder))


def __compute_score__(
    space: PredicateSpace[S],
    Qtable: QValuesLearner,
    sub_states: Set[S],
    depth_left: int,
    Qmax: Dict[S, int],
    classes: Dict[int, Set[S]],
) -> float:
    score = 0
    part_classes = {
        action: sum(Qtable[s][Qmax[s]] for s in sub_states)
        for action, sub in classes.items()
    }
    not_part_classes = {
        action: sum(Qtable[s][Qmax[s]] for s in space.seen if s not in sub_states)
        for action, sub in classes.items()
    }
    score = 0
    availables_pos = []
    availables_neg = []
    pli = sorted(list(part_classes.items()), key=lambda s: s[1], reverse=True)
    npli = sorted(list(not_part_classes.items()), key=lambda s: s[1], reverse=True)
    left = int(2 ** (depth_left - 1))
    while left > len(pli):
        left >>= 1
    for i in range(left):
        score += pli[i][1]
        score += npli[i][1]
        availables_pos.append(pli[i][0])
        availables_neg.append(npli[i][0])
    for s in space.seen:
        a = Qmax[s]
        if s in sub_states:
            if a not in availables_pos:
                score += max(Qtable[s][i] for i in availables_pos)
        else:
            if a not in availables_neg:
                score += max(Qtable[s][i] for i in availables_neg)
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
    best_score = __compute_score__(space, Qtable, space.seen, depth_left, Qmax, classes)

    for candidate, sub_states in space.predicates_set.items():

        score = __compute_score__(space, Qtable, sub_states, depth_left, Qmax, classes)
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
