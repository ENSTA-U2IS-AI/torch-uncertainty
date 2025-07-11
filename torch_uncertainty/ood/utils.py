from collections.abc import Iterator
from pathlib import Path
from typing import Any

import yaml


class ConfigNamespace:
    """Wrap a dict so you get BOTH attribute access (cfg.foo)
    and a dict API (cfg.keys(), cfg['foo'], cfg.get('foo')).
    """

    def __init__(self, d: dict):
        for k, v in d.items():
            setattr(self, k, _to_ns(v))

    # dict-style
    def keys(self) -> Iterator[str]:
        return self.__dict__.keys()

    def items(self):
        return self.__dict__.items()

    def values(self):
        return self.__dict__.values()

    def get(self, key, default=None):
        return self.__dict__.get(key, default)

    def __getitem__(self, key):
        """Allow dict-style access: cfg[key]."""
        return self.__dict__[key]

    def __repr__(self):
        """Return the canonical string representation."""
        return f"ConfigNamespace({self.__dict__!r})"


def _to_ns(obj: Any) -> Any:
    if isinstance(obj, dict):
        return ConfigNamespace(obj)
    if isinstance(obj, list):
        return [_to_ns(v) for v in obj]
    return obj


def load_config(path: str) -> ConfigNamespace:
    """Load any YAML file into a ConfigNamespace.

    You can then do cfg.foo.bar, cfg.foo.keys(), cfg['foo'], etc.
    """
    path = Path(path)
    with path.open() as f:
        raw = yaml.safe_load(f)
    return _to_ns(raw)
