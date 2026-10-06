"""Core types, registry and helpers shared by every method."""

from .registry import FAMILIES, MethodSpec, ParamSpec, get_method, list_methods, register, run
from .types import (
    Constraint,
    Dataset,
    LinearProgram,
    LinearSystem,
    Problem,
    Result,
    Step,
    as_vector,
    to_jsonable,
)

__all__ = [
    "FAMILIES",
    "Constraint",
    "Dataset",
    "LinearProgram",
    "LinearSystem",
    "MethodSpec",
    "ParamSpec",
    "Problem",
    "Result",
    "Step",
    "as_vector",
    "get_method",
    "list_methods",
    "register",
    "run",
    "to_jsonable",
]
