from collections import defaultdict
from typing import Callable, Dict, Generic, Iterable, TypeVar

from polext.decision_tree import DecisionTree

S = TypeVar("S")


class Forest(Generic[S]):
    def __init__(self, trees: Iterable[DecisionTree[S]]) -> None:
        self.trees = list(trees)

    def __call__(self, state: S) -> Dict[int, int]:
        votes = defaultdict(int)
        for tree in self.trees:
            votes[tree(state)] += 1
        return votes

    def policy(
        self, vote_decision: Callable[[Dict[int, int]], int]
    ) -> Callable[[S], int]:
        return lambda s: vote_decision(self(s))


def majority_vote(votes: Dict[int, int]) -> int:
    max_votes = -1
    maxi = 0
    for action, vote in votes.items():
        if vote > max_votes:
            max_votes = vote
            maxi = action
    return maxi


def vote_uncertainty(votes: Dict[int, int], choice: int) -> float:
    """
    Return the ratio of votes who did not vote for the choice
    """
    total = 0
    disagree = 0
    for action, vote in votes.items():
        total += vote
        if action != choice:
            disagree += vote
    return disagree / total
