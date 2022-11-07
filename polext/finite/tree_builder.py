from typing import Callable, Dict, List, Set, TypeVar
from polext.decision_tree import DecisionTree, Node, Leaf
from polext.predicate import Predicate

S = TypeVar("S")


def build_tree(
    states: List[S],
    Q: Callable[[S], List[float]],
    predicates: List[Predicate[S]],
    max_depth: int,
) -> DecisionTree[S]:
    Qtable = {s: Q(s) for s in states}
    predicates_table = {
        predicate: {s for s in states if predicate(s)} for predicate in predicates
    }
    nactions = len(Qtable[states[0]])
    return __rec_tree__(set(states), Qtable, predicates_table, max_depth, nactions, 0)


def __rec_tree__(
    states: Set[S],
    Qtable: Dict[S, List[float]],
    predicates_table: Dict[Predicate[S], Set[S]],
    depth_left: int,
    nactions: int,
    previous_action: int,
) -> DecisionTree[S]:
    if depth_left == 1:
        return Leaf(previous_action)
    # Find best split
    best_predicate = None
    best_action = previous_action
    best_score = sum(Qtable[s][previous_action] for s in states)
    for candidate, sub_states in predicates_table.items():
        part = sub_states.intersection(states)
        for action in range(nactions):
            score = sum(Qtable[s][action] for s in part)
            if score > best_score:
                best_predicate = candidate
                best_action = action
                best_score = score

    if best_predicate is None:
        return Leaf(previous_action)
    else:
        sub_states = predicates_table[best_predicate]
        next_pred_tables = {k: v for k, v in predicates_table.items()}
        del next_pred_tables[best_predicate]
        left = __rec_tree__(
            states.intersection(sub_states),
            Qtable,
            next_pred_tables,
            depth_left - 1,
            nactions,
            best_action,
        )
        right = __rec_tree__(
            states.difference(sub_states),
            Qtable,
            next_pred_tables,
            depth_left - 1,
            nactions,
            best_action,
        )
        return Node(best_predicate, left, right)
