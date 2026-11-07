"""Deterministic product tests; see README.md for the boundary rules."""

import platform
import sys

if sys.platform == "win32" and hasattr(platform, "_wmi_query"):
    # CPython's Windows fallback uses system version and processor environment values.
    # Install it before conftest imports numerical plugins, including in xdist workers.
    def _without_wmi(*_args, **_kwargs):
        raise OSError("WMI is unavailable in test processes")

    def _without_version_shell(system="", release="", version="", *_args, **_kwargs):
        return system, release, version

    platform._wmi_query = _without_wmi
    platform._syscmd_ver = _without_version_shell
