"""The method registry: one place that knows every method, its family and its parameters.

Methods register themselves with :func:`register`. The CLI, the fixture exporter and the
web app's parity tests all read the registry, so a method's id and parameter names are
its public contract.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Literal

from .types import Result, to_jsonable

Family = Literal[
    "roots",
    "systems",
    "scalar",
    "line_search",
    "unconstrained",
    "least_squares",
    "global",
    "stochastic",
    "constrained",
    "lp",
    "combinatorial",
    "linalg",
    "integration",
    "differentiation",
    "interpolation",
    "regression",
]

FAMILIES: tuple[str, ...] = Family.__args__  # type: ignore[attr-defined]

ParamKind = Literal["float", "int", "bool", "choice", "vector"]


@dataclass(frozen=True)
class ParamSpec:
    """A tunable parameter, described well enough for a UI to build a control for it.

    ``log=True`` asks for a logarithmic slider (tolerances, learning rates).
    """

    name: str
    default: Any
    kind: ParamKind = "float"
    min: float | None = None
    max: float | None = None
    choices: tuple[str, ...] = ()
    log: bool = False
    help: str = ""

    def to_dict(self) -> dict[str, Any]:
        return to_jsonable(
            {
                "name": self.name,
                "default": self.default,
                "kind": self.kind,
                "min": self.min,
                "max": self.max,
                "choices": list(self.choices),
                "log": self.log,
                "help": self.help,
            }
        )


@dataclass(frozen=True)
class MethodSpec:
    """Metadata for one registered method."""

    id: str
    family: str
    name: str
    fn: Callable[..., Result]
    params: tuple[ParamSpec, ...] = ()
    needs: tuple[str, ...] = ()
    order: str = ""
    summary: str = ""
    references: tuple[str, ...] = ()
    deterministic: bool = True
    tags: tuple[str, ...] = field(default_factory=tuple)

    def defaults(self) -> dict[str, Any]:
        return {p.name: p.default for p in self.params}

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "family": self.family,
            "name": self.name,
            "params": [p.to_dict() for p in self.params],
            "needs": list(self.needs),
            "order": self.order,
            "summary": self.summary,
            "references": list(self.references),
            "deterministic": self.deterministic,
            "tags": list(self.tags),
        }


_REGISTRY: dict[str, MethodSpec] = {}


def register(
    *,
    id: str,
    family: str,
    name: str,
    params: tuple[ParamSpec, ...] | list[ParamSpec] = (),
    needs: tuple[str, ...] = (),
    order: str = "",
    summary: str = "",
    references: tuple[str, ...] = (),
    deterministic: bool = True,
    tags: tuple[str, ...] = (),
) -> Callable[[Callable[..., Result]], Callable[..., Result]]:
    """Register ``fn`` under ``id``. The function signature must be ``fn(problem, **params)``."""
    if family not in FAMILIES:
        raise ValueError(f"unknown family {family!r}; expected one of {FAMILIES}")

    def decorator(fn: Callable[..., Result]) -> Callable[..., Result]:
        if id in _REGISTRY and _REGISTRY[id].fn is not fn:
            raise ValueError(f"method id {id!r} is already registered")
        _REGISTRY[id] = MethodSpec(
            id=id,
            family=family,
            name=name,
            fn=fn,
            params=tuple(params),
            needs=tuple(needs),
            order=order,
            summary=summary,
            references=tuple(references),
            deterministic=deterministic,
            tags=tuple(tags),
        )
        fn.spec = _REGISTRY[id]  # type: ignore[attr-defined]
        return fn

    return decorator


def _ensure_loaded() -> None:
    # Importing the package registers every method module.
    import numopt  # noqa: F401


def get_method(id: str) -> MethodSpec:
    _ensure_loaded()
    try:
        return _REGISTRY[id]
    except KeyError:
        raise KeyError(f"unknown method {id!r}; run `numopt list` to see all methods") from None


def list_methods(family: str | None = None) -> list[MethodSpec]:
    _ensure_loaded()
    specs = sorted(_REGISTRY.values(), key=lambda s: (FAMILIES.index(s.family), s.id))
    return [s for s in specs if family is None or s.family == family]


def run(method: str, problem: Any, **params: Any) -> Result:
    """Run a registered method by id: ``run("bfgs", problems.get("rosenbrock"), x0=[-1.2, 1])``."""
    spec = get_method(method)
    unknown = set(params) - {p.name for p in spec.params} - {"x0", "bracket", "seed"}
    if unknown:
        raise TypeError(f"{method}: unknown parameter(s) {sorted(unknown)}")
    return spec.fn(problem, **params)
