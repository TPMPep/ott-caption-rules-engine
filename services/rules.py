"""Immutable, job-scoped formatting rules.

Formatting jobs run concurrently in threads. Process-wide os.environ mutation can
therefore leak one project's spec into another. This module stores each job's
resolved caption rules in a ContextVar: immutable for the life of that job,
thread/task-local, and reset in the job runner's finally block.
"""
from contextvars import ContextVar, Token
from types import MappingProxyType
from typing import Any, Mapping, Optional
import os

_RULE_CONTEXT: ContextVar[Mapping[str, str]] = ContextVar(
    "caption_rule_context", default=MappingProxyType({})
)


def activate_rule_context(overrides: Optional[Mapping[str, Any]]) -> Token:
    normalized = {
        str(key).strip(): str(value)
        for key, value in (overrides or {}).items()
        if str(key).strip()
    }
    return _RULE_CONTEXT.set(MappingProxyType(normalized))


def reset_rule_context(token: Token) -> None:
    _RULE_CONTEXT.reset(token)


def get_rule(name: str, default: Optional[str] = None) -> Optional[str]:
    rules = _RULE_CONTEXT.get()
    if name in rules:
        return rules[name]
    return os.getenv(name, default)


def current_rules() -> Mapping[str, str]:
    return _RULE_CONTEXT.get()
