"""
Immutability primitive — a genuinely read-only mapping for value objects that
must NOT change after they are constructed.

WHY THIS EXISTS
───────────────
Several objects in the CC pipeline are conceptually IMMUTABLE and are shared BY
REFERENCE across many consumers:

  • the transcription-provenance object (assembly.build_transcription_provenance),
    stamped onto EVERY token in a run by shared reference; and
  • the unresolved-group object (segmentation.build_unresolved_group), forwarded
    verbatim by the formatter into Segmentation QC.

A plain Python ``dict`` shared by reference is MUTABLE. One accidental write
anywhere downstream (``token["transcription_provenance"]["provider"] = ...``)
would silently rewrite provenance for every token that references it — the
exact class of extremely-hard-to-diagnose corruption we refuse to depend on
"everyone honoring immutability" to avoid.

``FrozenDict`` makes that mutation IMPOSSIBLE, not merely discouraged: every
mutating operation raises ``TypeError`` at runtime. Correctness no longer
depends on every consumer being disciplined — the object physically cannot
change after construction.

DESIGN
──────
  • Read-only Mapping: implements the full ``collections.abc.Mapping`` protocol
    (``[]``, ``get``, ``in``, ``keys``, ``items``, ``values``, ``len``, iter),
    so every existing reader (``group.get("words")``, ``prov.get("provider")``,
    ``group["reason"]``) works UNCHANGED — this is a drop-in for the dicts these
    factories previously returned.
  • Deep freeze: nested dicts/lists are recursively converted to FrozenDict /
    tuples at construction, so ``prov["nested"]["x"] = 1`` is also blocked.
  • JSON/serialization safe: FrozenDict is a Mapping and ``freeze`` leaves
    JSON-native scalars untouched; a caller that must serialize can ``thaw()``
    to a fresh mutable copy WITHOUT touching the shared frozen original.
  • Hashable: frozen + hashable so a provenance/group object can be used in a
    set or as a dict key if ever needed (all values are frozen too).
  • Pure, dependency-free, deterministic. Mirrored on the JS side by
    ``Object.freeze`` (deep) in lib/cc-immutable.js for cross-language parity.

SOC 2 CC8.1 — an immutable value object is provably unchanged after its origin;
its integrity does not rely on downstream discipline.
"""

from typing import Any, Iterator, Mapping


class FrozenDict(Mapping):
    """A genuinely read-only mapping. Every mutating operation raises TypeError.

    Construct via ``freeze({...})`` (which deep-freezes) rather than calling this
    directly, so nested structures are frozen too. Reads behave exactly like a
    dict; writes are impossible.
    """

    __slots__ = ("_data", "_hash")

    def __init__(self, data: Mapping[str, Any]) -> None:
        # Store an already-frozen inner dict. ``freeze`` is responsible for
        # deep-freezing values before handing them here.
        object.__setattr__(self, "_data", dict(data))
        object.__setattr__(self, "_hash", None)

    # ── Mapping protocol (read-only) ─────────────────────────────────────────
    def __getitem__(self, key: str) -> Any:
        return self._data[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)

    def __contains__(self, key: object) -> bool:
        return key in self._data

    def __repr__(self) -> str:
        return f"FrozenDict({self._data!r})"

    def __eq__(self, other: object) -> bool:
        if isinstance(other, FrozenDict):
            return self._data == other._data
        if isinstance(other, Mapping):
            return dict(self._data) == dict(other)
        return NotImplemented

    def __hash__(self) -> int:
        if self._hash is None:
            object.__setattr__(self, "_hash", hash(frozenset(self._data.items())))
        return self._hash

    # ── Mutation guards — every path raises ──────────────────────────────────
    def __setitem__(self, key: str, value: Any) -> None:
        raise TypeError("FrozenDict is immutable; cannot set item")

    def __delitem__(self, key: str) -> None:
        raise TypeError("FrozenDict is immutable; cannot delete item")

    def __setattr__(self, name: str, value: Any) -> None:
        raise TypeError("FrozenDict is immutable; cannot set attribute")

    def __delattr__(self, name: str) -> None:
        raise TypeError("FrozenDict is immutable; cannot delete attribute")

    def clear(self) -> None:
        raise TypeError("FrozenDict is immutable; cannot clear")

    def pop(self, *args: Any, **kwargs: Any) -> Any:
        raise TypeError("FrozenDict is immutable; cannot pop")

    def popitem(self) -> Any:
        raise TypeError("FrozenDict is immutable; cannot popitem")

    def setdefault(self, *args: Any, **kwargs: Any) -> Any:
        raise TypeError("FrozenDict is immutable; cannot setdefault")

    def update(self, *args: Any, **kwargs: Any) -> None:
        raise TypeError("FrozenDict is immutable; cannot update")


def freeze(value: Any) -> Any:
    """Deep-freeze a JSON-native value into an immutable form.

    dict → FrozenDict (recursively), list/tuple → tuple of frozen items, and
    JSON scalars (str/int/float/bool/None) pass through unchanged. The result is
    a genuinely read-only object: any attempt to mutate it (or any nested part
    of it) raises TypeError. Deterministic; safe on already-frozen inputs.
    """
    if isinstance(value, FrozenDict):
        return value
    if isinstance(value, Mapping):
        return FrozenDict({k: freeze(v) for k, v in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(freeze(v) for v in value)
    return value


def thaw(value: Any) -> Any:
    """Deep-copy a (possibly frozen) value into a fresh MUTABLE structure.

    FrozenDict → dict (recursively), tuple → list, scalars unchanged. Use this
    only when a mutable copy is genuinely required (e.g. before serialization
    that mutates in place); it NEVER touches the shared frozen original, so the
    immutability guarantee is preserved.
    """
    if isinstance(value, (FrozenDict, Mapping)):
        return {k: thaw(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [thaw(v) for v in value]
    return value
