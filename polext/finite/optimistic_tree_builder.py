from typing import Dict, List, Set, TypeVar

import numpy as np

from polext.decision_tree import DecisionTree, Node
from polext.finite.greedy_builder import greedy_finisher, greedy_q_selection
from polext.predicate import Predicate
from polext.predicate_space import PredicateSpace

S = TypeVar("S")


def optimistic_tree_builder(loss):
    def f(space: PredicateSpace[S], max_depth: int, **kwargs) -> DecisionTree[S]:
        Qmax = {s: np.argmax(vals) for s, vals in space.Qtable.items()}
        classes = {action: set() for action in range(space.nactions)}
        for s in space.states:
            a = Qmax[s]
            classes[a].add(s)
        return __builder__(
            space,
            max_depth,
            Qmax,
            loss,
            classes,
        )

    return f


def __compute_score__(
    states: Set[S],
    sub_states: Set[S],
    Qtable: Dict[S, List[float]],
    depth_left: int,
    Qmax: Dict[S, int],
    classes: Dict[int, Set[S]],
) -> float:
    score = 0
    part_classes = {
        action: sum(Qtable[s][Qmax[s]] for s in sub.intersection(sub_states))
        for action, sub in classes.items()
    }
    not_part_classes = {
        action: sum(Qtable[s][Qmax[s]] for s in sub.difference(sub_states))
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
    for s in states:
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
    depth_left: int,
    Qmax: Dict[S, int],
    loss,
    classes: Dict[int, Set[S]],
) -> DecisionTree[S]:
    if depth_left <= 2:
        tree = greedy_finisher(space, greedy_q_selection, loss)
        assert tree
        return tree

    # Compute current score
    best_predicate = None
    best_score = __compute_score__(
        space.states, space.states, space.Qtable, 1, Qmax, classes
    )

    for candidate, sub_states in space.predicates_set.items():

        score = __compute_score__(
            space.states, sub_states, space.Qtable, depth_left, Qmax, classes
        )
        if score > best_score:
            best_score = score
            best_predicate = candidate

    if best_predicate is None:
        tree = greedy_finisher(space, greedy_q_selection, loss)
        assert tree
        return tree
    sub_states = space.predicates_set[best_predicate]
    left_space, right_space = space.children(best_predicate)

    left = __builder__(
        left_space,
        depth_left - 1,
        Qmax,
        loss,
        {action: sub.intersection(sub_states) for action, sub in classes.items()},
    )
    right = __builder__(
        right_space,
        depth_left - 1,
        Qmax,
        loss,
        {action: sub.difference(sub_states) for action, sub in classes.items()},
    )
    return Node(best_predicate, left, right)
