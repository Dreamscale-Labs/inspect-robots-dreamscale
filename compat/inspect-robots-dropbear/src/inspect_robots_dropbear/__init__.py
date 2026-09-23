"""Renamed to `inspect_robots_dreamscale`; this name keeps existing imports working."""

import sys

from inspect_robots_dreamscale import (
    DreamscalePolicy,
    DropbearPolicy,
    dreamscale_policy,
    dreamzero_yam,
    dropbear_policy,
    policy,
    telemetry,
)

# `from inspect_robots_dropbear.policy import DropbearPolicy` and friends resolve
# to the same module objects as the new names, so patches apply to both.
for _name, _module in (
    ("dreamzero_yam", dreamzero_yam),
    ("policy", policy),
    ("telemetry", telemetry),
):
    sys.modules[f"{__name__}.{_name}"] = _module

__all__ = ["DreamscalePolicy", "DropbearPolicy", "dreamscale_policy", "dropbear_policy"]
