"""Auto registry for tools defined in modules named tools_*.py."""

from __future__ import annotations

import importlib
import pkgutil
from types import ModuleType


def _iter_tool_modules() -> list[ModuleType]:
    """Discover and import all sibling modules matching tools_*.py."""
    modules: list[ModuleType] = []
    package_name = __name__.rsplit(".", 1)[0]
    package = importlib.import_module(package_name)
    package_path = getattr(package, "__path__", None)
    if package_path is None:
        return modules

    for module_info in pkgutil.iter_modules(package_path):
        name = module_info.name
        if not name.startswith("tools_"):
            continue
        full_name = f"{package_name}.{name}"
        modules.append(importlib.import_module(full_name))

    modules.sort(key=lambda mod: mod.__name__)
    return modules


def _export_module_symbols(module: ModuleType) -> list[str]:
    """Export symbols from a tool module into this registry module."""
    names = getattr(module, "__all__", None)
    if names is None:
        names = [n for n in vars(module).keys() if not n.startswith("_")]

    exported: list[str] = []
    for name in names:
        globals()[name] = getattr(module, name)
        exported.append(name)
    return exported


def _bootstrap_registry() -> list[str]:
    exported: list[str] = []
    for module in _iter_tool_modules():
        exported.extend(_export_module_symbols(module))
    deduped = sorted(set(exported))
    return deduped


__all__ = _bootstrap_registry()
