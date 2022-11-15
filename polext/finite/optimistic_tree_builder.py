from typing import Dict, List, Set, TypeVar

from polext.decision_tree import DecisionTree, Node
from polext.predicate import Predicate

from polext.finite.greedy_builder import greedy_finisher, greedy_q_selection

S = TypeVar("S")


def optimistic_tree_builder(loss):
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
    best_score = __compute_score__(states, states, Qtable, 1, Qmax, classes)

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
