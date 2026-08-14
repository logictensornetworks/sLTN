from abc import ABC, abstractmethod


class ConnectiveOperator(ABC):
    """Abstract base for all fuzzy connective operators (unary and binary).

    Subclass :class:`UnaryConnectiveOperator` or :class:`BinaryConnectiveOperator`
    for concrete operator implementations.  This class exists so that
    ``isinstance(op, ConnectiveOperator)`` and subclassing work correctly.
    """

    @abstractmethod
    def __repr__(self) -> str: ...


class UnaryConnectiveOperator(ConnectiveOperator):
    """Base class for unary fuzzy connectives (e.g. negation)."""

    @abstractmethod
    def __call__(self, x, **kwargs): ...


class BinaryConnectiveOperator(ConnectiveOperator):
    """Base class for binary fuzzy connectives (e.g. conjunction, disjunction, implication)."""

    @abstractmethod
    def __call__(self, x, y, **kwargs): ...


class AggregationOperator(ABC):
    """Base class for fuzzy aggregation operators used by quantifiers."""

    @abstractmethod
    def __call__(self, *args, **kwargs): ...

    @abstractmethod
    def __repr__(self) -> str: ...
