from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Generic, TypeVar

from polext.predicate import Predicate

S = TypeVar("S")


class DecisionTree(ABC, Generic[S]):
    @abstractmethod
    def __call__(self, state: S) -> int:
        pass

    @abstractmethod
    def to_string(self, level: int = 0) -> str:
        pass


@dataclass
class Node(DecisionTree[S]):
    predicate: Predicate[S]
    left: DecisionTree[S]
    right: DecisionTree[S]

    def __call__(self, state: S) -> int:
        if self.predicate(state):
            return self.left(state)
        return self.right(state)

    def to_string(self, level: int = 0) -> str:
        return (
            "\t" * level
            + f"{self.predicate}:\n{self.left.to_string(level + 1)}\n{self.right.to_string(level + 1)}"
        )

    def __repr__(self) -> str:
        return self.to_string()


@dataclass
class Leaf(DecisionTree[S]):
    action: int

    def __call__(self, state: S) -> int:
        return self.action

    def to_string(self, level: int = 0) -> str:
        return "\t" * level + f"{self.action}"

    def __repr__(self) -> str:
        return self.to_string()
