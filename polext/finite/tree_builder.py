from typing import Callable, Dict, Iterable, List, Optional, Set, Tuple, TypeVar
from polext.decision_tree import DecisionTree, Node, Leaf
from polext.predicate import Predicate

S = TypeVar("S")


def __loss__(
    tree: DecisionTree[S], Qtable: Dict[S, List[float]], states: Iterable[S]
) -> float:
    lost_reward = sum(max(Qtable[s]) - Qtable[s][tree(s)] for s in states)
    return lost_reward


def build_tree(
    states: List[S],
    Q: Callable[[S], List[float]],
    predicates: List[Predicate[S]],
    max_depth: int,
    method: str,
) -> Tuple[DecisionTree[S], float]:
    Qtable = {s: Q(s) for s in states}
    predicates_table = {
        predicate: {s for s in states if predicate(s)} for predicate in predicates
    }
    nactions = len(Qtable[states[0]])
    best_loss = 1e99
    best = Leaf(0)
    for action in range(nactions):
        tree = __rec_tree__(
            set(states), Qtable, predicates_table, max_depth, nactions, action, method
        ).simplified()
        lost_reward = __loss__(tree, Qtable, states)
        if lost_reward < best_loss:
            best = tree
            best_loss = lost_reward

    return best, best_loss


def __split__(
    best_predicate: Optional[Predicate[S]],
    best_action: int,
    states: Set[S],
    Qtable: Dict[S, List[float]],
    predicates_table: Dict[Predicate[S], Set[S]],
    depth_left: int,
    nactions: int,
    previous_action: int,
    method: str,
) -> DecisionTree[S]:
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
            method,
        )
        right = __rec_tree__(
            states.difference(sub_states),
            Qtable,
            next_pred_tables,
            depth_left - 1,
            nactions,
            previous_action,
            method,
        )
        return Node(best_predicate, left, right)


def __greedy_q_selection__(
    states: Set[S],
    Qtable: Dict[S, List[float]],
    predicates_table: Dict[Predicate[S], Set[S]],
    depth_left: int,
    nactions: int,
    previous_action: int,
) -> Tuple[Optional[Predicate[S]], int, int]:
    best_predicate = None
    best_action = previous_action
    best_score = sum(Qtable[s][previous_action] for s in states)
    for candidate, sub_states in predicates_table.items():
        # print("\tcandidate:", candidate)
        part = states.intersection(sub_states)
        not_part = states.difference(sub_states)
        free_score = sum(Qtable[s][previous_action] for s in not_part)
        for action in range(nactions):
            if action == previous_action:
                continue
            score = sum(Qtable[s][action] for s in part) + free_score
            # print("\t\taction:", action, "score:", score)
            if score > best_score:
                best_predicate = candidate
                best_action = action
                best_score = score
    return best_predicate, best_action, previous_action


def __greedy_opt_action_selection__(
    states: Set[S],
    Qtable: Dict[S, List[float]],
    predicates_table: Dict[Predicate[S], Set[S]],
    depth_left: int,
    nactions: int,
    previous_action: int,
) -> Tuple[Optional[Predicate[S]], int, int]:
    best_predicate = None
    best_action = previous_action
    n_before = sum(1 for s in states if max(Qtable[s]) <= Qtable[s][previous_action])
    best_score = n_before
    for candidate, sub_states in predicates_table.items():
        # print("\tcandidate:", candidate)
        part = states.intersection(sub_states)
        not_part = states.difference(sub_states)
        n_candidate = sum(
            1 for s in not_part if max(Qtable[s]) <= Qtable[s][previous_action]
        )
        for action in range(nactions):
            if action == previous_action:
                continue
            n_after = (
                sum(1 for s in part if max(Qtable[s]) <= Qtable[s][action])
                + n_candidate
            )
            score = n_after
            # print("\t\taction:", action, "score:", score)
            if score > best_score:
                best_predicate = candidate
                best_action = action
                best_score = score
    return best_predicate, best_action, previous_action


def __find_greedy_finisher__(
    states: Set[S],
    Qtable: Dict[S, List[float]],
    predicates_table: Dict[Predicate[S], Set[S]],
    nactions: int,
    fun,
) -> Tuple[Optional[Predicate[S]], int, int]:
    best_score = 1e99
    answer = (None, 0, 0)
    for action in range(nactions):
        pred, baction, _ = fun(states, Qtable, predicates_table, 2, nactions, action)
        tree = Node(pred, Leaf(baction), Leaf(action)) if pred else Leaf(action)
        score = __loss__(tree, Qtable, states)
        if score < best_score:
            best_score = score
            answer = pred, baction, action
    return answer


def __best_probability_selection__(
    states: Set[S],
    Qtable: Dict[S, List[float]],
    predicates_table: Dict[Predicate[S], Set[S]],
    depth_left: int,
    nactions: int,
    previous_action: int,
) -> Tuple[Optional[Predicate[S]], int, int]:
    best_score = -999999
    if depth_left == 2:
        return __find_greedy_finisher__(
            states, Qtable, predicates_table, nactions, __greedy_q_selection__
        )

    best_predicate = None
    best_action = previous_action
    best_score = 0

    classes = {action: 0 for action in range(nactions)}
    for s in states:
        best_action = Qtable[s].index(max(Qtable[s]))
        if isinstance(best_action, int):
            best_action = (best_action,)
        for a in best_action:
            classes[a] += 1
    tpart = sum(classes.values())
    for s in states:
        best_action = Qtable[s].index(max(Qtable[s]))
        if isinstance(best_action, int):
            best_action = (best_action,)
        best_score += min(1, sum(classes[a] for a in best_action) / tpart) * max(
            Qtable[s]
        )

    for candidate, sub_states in predicates_table.items():
        # print("\tcandidate:", candidate)
        part_classes = {action: 0 for action in range(nactions)}
        not_part_classes = {action: 0 for action in range(nactions)}
        for s in states:
            best_action = Qtable[s].index(max(Qtable[s]))
            if isinstance(best_action, int):
                best_action = (best_action,)
            if s in sub_states:
                for a in best_action:
                    part_classes[a] += 1
            else:
                for a in best_action:
                    not_part_classes[a] += 1
        score = 0
        tpart = sum(part_classes.values())
        tnpart = sum(not_part_classes.values())
        for s in states:
            best_action = Qtable[s].index(max(Qtable[s]))
            if isinstance(best_action, int):
                best_action = (best_action,)
            if s in sub_states:
                score += min(
                    1, sum(part_classes[a] for a in best_action) / tpart
                ) * max(Qtable[s])
            else:
                score += min(
                    1, sum(not_part_classes[a] for a in best_action) / tnpart
                ) * max(Qtable[s])

        if score > best_score:
            best_score = score
            best_predicate = candidate

    if best_predicate is None:
        return __find_greedy_finisher__(
            states, Qtable, predicates_table, nactions, __greedy_q_selection__
        )
    return best_predicate, previous_action, previous_action


METHODS = {
    "greedy-q": __greedy_q_selection__,
    "greedy-opt-action": __greedy_opt_action_selection__,
    "max-probability": __best_probability_selection__,
}


def __rec_tree__(
    states: Set[S],
    Qtable: Dict[S, List[float]],
    predicates_table: Dict[Predicate[S], Set[S]],
    depth_left: int,
    nactions: int,
    previous_action: int,
    method: str,
) -> DecisionTree[S]:
    if depth_left == 1:
        return Leaf(previous_action)
    # Find best split
    best_predicate, best_action, previous_action = METHODS[method](
        states, Qtable, predicates_table, depth_left, nactions, previous_action
    )

    return __split__(
        best_predicate,
        best_action,
        states,
        Qtable,
        predicates_table,
        depth_left,
        nactions,
        previous_action,
        method,
    )
