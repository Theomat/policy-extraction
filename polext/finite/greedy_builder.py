from typing import Dict, List, Optional, Set, Tuple, TypeVar
from polext.decision_tree import DecisionTree, Node, Leaf
from polext.predicate import Predicate

S = TypeVar("S")


def greedy_finisher(
    states: Set[S],
    Qtable: Dict[S, List[float]],
    predicates_table: Dict[Predicate[S], Set[S]],
    nactions: int,
    fun,
    loss,
    **kwargs
) -> Optional[DecisionTree[S]]:
    best_score = 1e99
    answer = None
    for action in range(nactions):
        pred, baction, _ = fun(
            states, Qtable, predicates_table, 2, nactions, action, **kwargs
        )
        tree = Node(pred, Leaf(baction), Leaf(action)) if pred else Leaf(action)
        score = loss(tree, Qtable, states)
        if score < best_score:
            best_score = score
            answer = tree
    return answer


def greedy_tree_builder(fn, loss):
    def f(
        states: Set[S],
        Qtable: Dict[S, List[float]],
        predicates_table: Dict[Predicate[S], Set[S]],
        nactions: int,
        max_depth: int,
        **kwargs
    ) -> DecisionTree[S]:
        best = Leaf(0)
        best_loss = 1e99
        for action in range(nactions):
            tree = __rec_tree__(
                fn,
                set(states),
                Qtable,
                predicates_table,
                max_depth,
                nactions,
                action,
            ).simplified()
            lost_reward = loss(tree, Qtable, states)
            if lost_reward < best_loss:
                best = tree
                best_loss = lost_reward
        return best

    return f


def __split__(
    fn,
    best_predicate: Optional[Predicate[S]],
    best_action: int,
    states: Set[S],
    Qtable: Dict[S, List[float]],
    predicates_table: Dict[Predicate[S], Set[S]],
    depth_left: int,
    nactions: int,
    previous_action: int,
    **kwargs
) -> DecisionTree[S]:
    if best_predicate is None:
        return Leaf(previous_action)
    else:
        sub_states = predicates_table[best_predicate]
        next_pred_tables = {k: v for k, v in predicates_table.items()}
        del next_pred_tables[best_predicate]
        left = __rec_tree__(
            fn,
            states.intersection(sub_states),
            Qtable,
            next_pred_tables,
            depth_left - 1,
            nactions,
            best_action,
            **kwargs
        )
        right = __rec_tree__(
            fn,
            states.difference(sub_states),
            Qtable,
            next_pred_tables,
            depth_left - 1,
            nactions,
            previous_action,
            **kwargs
        )
        return Node(best_predicate, left, right)


def __rec_tree__(
    fn,
    states: Set[S],
    Qtable: Dict[S, List[float]],
    predicates_table: Dict[Predicate[S], Set[S]],
    depth_left: int,
    nactions: int,
    previous_action: int,
    **kwargs
) -> DecisionTree[S]:
    if depth_left == 1:
        return Leaf(previous_action)
    # Find best split
    best_predicate, best_action, previous_action = fn(
        states,
        Qtable,
        predicates_table,
        depth_left,
        nactions,
        previous_action,
        **kwargs
    )

    return __split__(
        fn,
        best_predicate,
        best_action,
        states,
        Qtable,
        predicates_table,
        depth_left,
        nactions,
        previous_action,
        **kwargs
    )


def greedy_q_selection(
    states: Set[S],
    Qtable: Dict[S, List[float]],
    predicates_table: Dict[Predicate[S], Set[S]],
    depth_left: int,
    nactions: int,
    previous_action: int,
    **kwargs
) -> Tuple[Optional[Predicate[S]], int, int]:
    best_predicate = None
    best_action = previous_action
    best_score = sum(Qtable[s][previous_action] for s in states)
    for candidate, sub_states in predicates_table.items():
        part = states.intersection(sub_states)
        not_part = states.difference(sub_states)
        free_score = sum(Qtable[s][previous_action] for s in not_part)
        for action in range(nactions):
            if action == previous_action:
                continue
            score = sum(Qtable[s][action] for s in part) + free_score
            if score > best_score:
                best_predicate = candidate
                best_action = action
                best_score = score
    return best_predicate, best_action, previous_action


def greedy_opt_action_selection(
    states: Set[S],
    Qtable: Dict[S, List[float]],
    predicates_table: Dict[Predicate[S], Set[S]],
    depth_left: int,
    nactions: int,
    previous_action: int,
    **kwargs
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
