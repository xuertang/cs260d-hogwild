"""Training modes. Each module exposes one class implementing schema.Mode.

    serial   → modes/serial.py   stream B  (not yet created)
    sync     → modes/sync.py     stream C
    backup   → modes/backup.py   stream C
    hogwild  → modes/hogwild.py  stream A
"""

from __future__ import annotations

from importlib import import_module

from hogwild.schema import Mode

# mode name → (module, class). Imported lazily so that one missing or broken
# mode does not stop the others from running while streams land separately.
_REGISTRY: dict[str, tuple[str, str]] = {
    "serial": ("hogwild.modes.serial", "SerialMode"),
    "sync": ("hogwild.modes.sync", "SyncMode"),
    "backup": ("hogwild.modes.backup", "BackupMode"),
    "hogwild": ("hogwild.modes.hogwild", "HogwildMode"),
}


def get_mode(name: str) -> Mode:
    """Instantiate the Mode for Config.mode."""
    try:
        module_name, class_name = _REGISTRY[name]
    except KeyError:
        raise ValueError(f"unknown mode {name!r}; expected one of {sorted(_REGISTRY)}") from None
    return getattr(import_module(module_name), class_name)()
