from typing import Dict, List, Set, TypeVar

from polext.decision_tree import DecisionTree, Node
from polext.predicate import Predicate

from polext.finite.greedy_builder import greedy_finisher, greedy_q_selection

S = TypeVar("S")


def max_probability_tree_builder(loss):
    def f(
        states: Set[S],
        Qtable: Dict[S, List[float]],
        predicates_table: Dict[Predicate[S], Set[S]],
        nactions: int,
        max_depth: int,
        **kwargs
    ) -> DecisionTree[S]:
        Qmax = {s: vals.index(max(vals)) for s, vals in Qtable.items()}
        classes = {action: set() for action in range(nactions)}
        for s in states:
            a = Qmax[s]
            classes[a].add(s)

        return __builder__(
            states, Qtable, predicates_table, nactions, max_depth, Qmax, loss, classes
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
        action: len(s.intersection(sub_states)) for action, s in classes.items()
    }
    not_part_classes = {
        action: len(s.difference(sub_states)) for action, s in classes.items()
    }
    tpart = sum(part_classes.values())
    tnpart = sum(not_part_classes.values())
    for s in states:
        a = Qmax[s]
        if s in sub_states:
            score += part_classes[a] / tpart * Qtable[s][Qmax[s]]
        else:
            score += not_part_classes[a] / tnpart * Qtable[s][Qmax[s]]
    return score


def __builder__(
    states: Set[S],
    Qtable: Dict[S, List[float]],
    predicates_table: Dict[Predicate[S], Set[S]],
    nactions: int,
    depth_left: int,
    Qmax: Dict[S, int],
    loss,
    classes: Dict[int, Set[S]],
) -> DecisionTree[S]:
    if depth_left <= 2:
        tree = greedy_finisher(
            states, Qtable, predicates_table, nactions, greedy_q_selection, loss
        )
        assert tree
        return tree

    # Compute current score
    best_predicate = None
    best_score = __compute_score__(states, states, Qtable, depth_left, Qmax, classes)

    for candidate, sub_states in predicates_table.items():

        score = __compute_score__(states, sub_states, Qtable, depth_left, Qmax, classes)
        if score > best_score:
            best_score = score
            best_predicate = candidate

    if best_predicate is None:
        tree = greedy_finisher(
            states, Qtable, predicates_table, nactions, greedy_q_selection, loss
        )
        assert tree
        return tree
    sub_states = predicates_table[best_predicate]
    next_pred_tables = {k: v for k, v in predicates_table.items()}
    del next_pred_tables[best_predicate]
    left = __builder__(
        states.intersection(sub_states),
        Qtable,
        next_pred_tables,
        nactions,
        depth_left - 1,
        Qmax,
        loss,
        {action: sub.intersection(sub_states) for action, sub in classes.items()},
    )
    right = __builder__(
        states.difference(sub_states),
        Qtable,
        next_pred_tables,
        nactions,
        depth_left - 1,
        Qmax,
        loss,
        {action: sub.difference(sub_states) for action, sub in classes.items()},
    )
    return Node(best_predicate, left, right)
