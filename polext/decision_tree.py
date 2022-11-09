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

    def simplified(self) -> "DecisionTree[S]":
        return self


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

    def simplified(self) -> "DecisionTree[S]":
        if (
            isinstance(self.left, Leaf)
            and isinstance(self.right, Leaf)
            and self.left.action == self.right.action
        ):
            return self.left
        sleft = self.left.simplified()
        sright = self.right.simplified()
        if sleft != self.left or sright != self.right:

            return Node(self.predicate, sleft, sright).simplified()
        else:
            return Node(self.predicate, sleft, sright)


@dataclass
class Leaf(DecisionTree[S]):
    action: int

    def __call__(self, state: S) -> int:
        return self.action

    def to_string(self, level: int = 0) -> str:
        return "\t" * level + f"{self.action}"

    def __repr__(self) -> str:
        return self.to_string()
