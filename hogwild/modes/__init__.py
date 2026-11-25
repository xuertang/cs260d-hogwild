"""Training modes. Each module exposes one class implementing schema.Mode.

    serial   → modes/serial.py    one worker; the reference curve
    sync     → modes/sync.py      m workers, barrier, averaged gradient
    backup   → modes/backup.py    m + b workers, first m gradients win
    hogwild  → modes/hogwild.py   m workers, lock-free in-place updates
"""

from __future__ import annotations

from importlib import import_module

from hogwild.schema import Mode

# mode name → (module, class). Imported lazily: runner.py imports this package
# and the mode modules import runner.py.
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
