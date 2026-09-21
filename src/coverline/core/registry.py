"""The registry, and why the type annotation on it is the actual enforcement.

The drift failure -- the same job done slightly differently per league, so a
fix in one never reaches the others -- is not fixed by writing five careful
implementations. It is fixed by making the five implementations answer to one
type, and making the type checker refuse the ones that do not.

The mechanism is narrow and worth stating exactly, because the obvious
alternative does not work:

    @runtime_checkable + isinstance(obj, LeagueModel)   # DOES NOT CHECK SIGNATURES

The typing documentation is explicit that runtime protocol checks verify only
the PRESENCE of members, not their signatures. A league whose predict() takes
the wrong arguments passes isinstance and fails in production. What does work
is assignment into a dict annotated with the protocol: mypy checks the
assignment structurally, at the @register call site, including signatures.

So the rule for this package is:

    * REGISTRY is annotated dict[str, LeagueModel]
    * every league registers through @register
    * CI runs mypy --strict over src/

and a league missing a method, or typing one wrong, fails CI at the decorator
rather than at 11:00 UTC on a Tuesday.
"""

from __future__ import annotations

from typing import Callable, TypeVar

from coverline.core.interfaces import LeagueModel

#: The single source of truth for which leagues exist.
#: The annotation is load-bearing -- see module docstring.
REGISTRY: dict[str, LeagueModel] = {}

#: Leagues this project intends to support. A name here with no entry in
#: REGISTRY is an unbuilt league; a name in REGISTRY that is not here is a
#: league someone added without deciding it was in scope. The conformance
#: guard tests assert on both directions, so neither can happen quietly.
EXPECTED_LEAGUES: frozenset[str] = frozenset({"nfl", "cfb", "mlb", "nhl", "nba"})

T = TypeVar("T", bound=LeagueModel)


def register(model: T) -> T:
    """Register a league model instance. Returns it unchanged so it can decorate.

    Raises on duplicate registration rather than overwriting: two modules
    claiming the same league is a merge accident, and silently keeping the
    last one is how the wrong model ends up on the board.
    """
    league = model.league
    if league in REGISTRY:
        raise ValueError(
            f"league {league!r} is already registered by "
            f"{type(REGISTRY[league]).__module__}; refusing to overwrite"
        )
    if league not in EXPECTED_LEAGUES:
        raise ValueError(
            f"league {league!r} is not in EXPECTED_LEAGUES {sorted(EXPECTED_LEAGUES)}; "
            "add it there deliberately before registering"
        )
    REGISTRY[league] = model
    return model


def get(league: str) -> LeagueModel:
    """Look up a registered league, with a message that says what exists."""
    try:
        return REGISTRY[league]
    except KeyError:
        raise KeyError(
            f"no league registered as {league!r}; registered: {sorted(REGISTRY)}"
        ) from None


def registered() -> frozenset[str]:
    return frozenset(REGISTRY)


def missing() -> frozenset[str]:
    """Leagues declared in scope but not yet built. Printed by the run manifest."""
    return EXPECTED_LEAGUES - registered()


def clear_for_testing() -> None:
    """Empty the registry. Only tests should call this."""
    REGISTRY.clear()
