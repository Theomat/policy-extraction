from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Generic, Optional, TypeVar

from polext.predicate import Predicate

from rich import print
from rich.tree import Tree
from rich.text import Text

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

    def print(self, parent: Optional[Tree] = None):
        pass

    def size(self) -> int:
        return 1


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

    def print(self, parent: Optional[Tree] = None):
        pred_text = Text.assemble((f"{self.predicate}", "bright_blue"))
        first = parent is None
        if first:
            size = self.size()
            if size > 50:
                print(
                    Text.assemble(
                        (
                            f"Tree of size {size} is too big for the console...",
                            "italic red",
                        )
                    )
                )
                return
            parent = Tree(pred_text)
        else:
            parent = parent.add(pred_text)
        self.left.print(parent)
        self.right.print(parent)
        if first:
            print(parent)

    def size(self) -> int:
        return 1 + self.left.size() + self.right.size()


@dataclass
class Leaf(DecisionTree[S]):
    action: int

    def __call__(self, state: S) -> int:
        return self.action

    def to_string(self, level: int = 0) -> str:
        return "\t" * level + f"{self.action}"

    def __repr__(self) -> str:
        return self.to_string()

    def print(self, parent: Optional[Tree] = None):
        action = Text.assemble((f"{self.action}", "bright_green"))
        if parent is None:
            parent = Tree(action)
            print(parent)
        else:
            parent = parent.add(action)
