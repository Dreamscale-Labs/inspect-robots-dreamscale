"""Strict conversion from Inspect observations to Dreamscale DreamZero-YAM input.

Capture timing has two sources, and every converted observation says which one
it used:

``per_camera``
    The embodiment filled ``Observation.image_times`` for all three cameras with
    real Unix-epoch capture seconds (the Dreamscale YAM fork does this). These
    are validated strictly: a time more than ``MAX_CAPTURE_AGE_S`` old or more
    than ``MAX_CAPTURE_FUTURE_S`` ahead of the wall clock is rejected.

``observation_fallback``
    The embodiment left ``image_times`` empty, as stock upstream
    ``inspect-robots-yam`` does (it never sets ``image_times`` or
    ``state_time``). All three cameras are then stamped with one time: the
    adapter's wall clock (``time.time()``) at the moment ``act()`` received the
    observation. That is later than the true capture by the embodiment's
    capture-to-return latency, so frame age is under-reported and a stalled
    camera cannot be detected here. DreamZero-YAM's server admits frames by
    control tick, not by capture time, and only requires capture times that
    never move backwards and precede encode; both hold for this stamp.

A *partial* ``image_times`` (some cameras stamped, some not) is still rejected:
that is an embodiment bug, not a different contract.
"""

from __future__ import annotations

import math
import time
from collections.abc import Mapping
from typing import Any, Literal, cast

import dreamscale as _dreamscale  # type: ignore[import-untyped]
import numpy as np
import numpy.typing as npt
from dreamscale.dreamzero_yam import DreamZeroYamObservation  # type: ignore[import-untyped]
from inspect_robots.types import Observation

_CAMERA_NAMES = ("top_cam", "left_cam", "right_cam")
MAX_CAPTURE_AGE_S = 5.0
MAX_CAPTURE_FUTURE_S = 1.0
CaptureTiming = Literal["per_camera", "observation_fallback"]
PER_CAMERA: CaptureTiming = "per_camera"
OBSERVATION_FALLBACK: CaptureTiming = "observation_fallback"
dreamscale: Any = _dreamscale


def _seconds_to_ns(value: object, *, camera: str) -> int:
    """Convert finite Unix-epoch capture seconds to integer nanoseconds."""
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"image_times[{camera!r}] must be finite Unix-epoch seconds")
    try:
        seconds = float(cast(Any, value))
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"image_times[{camera!r}] must be finite Unix-epoch seconds"
        ) from error
    if not math.isfinite(seconds) or seconds < 0.0:
        raise ValueError(f"image_times[{camera!r}] must be finite Unix-epoch seconds")
    return round(seconds * 1_000_000_000)


def _capture_times_ns(observation: Observation) -> tuple[int, int, int]:
    """Validate source capture times before any inference request is possible."""
    now_s = time.time()
    times_ns: list[int] = []
    for name in _CAMERA_NAMES:
        raw = _mapping_value(observation.image_times, name, field="image_times")
        converted = _seconds_to_ns(raw, camera=name)
        capture_s = converted / 1_000_000_000
        if capture_s < now_s - MAX_CAPTURE_AGE_S:
            raise ValueError(
                f"image_times[{name!r}] is stale; source capture time must be within "
                f"{MAX_CAPTURE_AGE_S:g} seconds"
            )
        if capture_s > now_s + MAX_CAPTURE_FUTURE_S:
            raise ValueError(
                f"image_times[{name!r}] is implausibly future; source capture time "
                f"must not exceed wall clock by more than {MAX_CAPTURE_FUTURE_S:g} second"
            )
        times_ns.append(converted)
    return cast(tuple[int, int, int], tuple(times_ns))


def _fallback_capture_times_ns(received_s: float | None) -> tuple[int, int, int]:
    """Stamp every camera with the adapter's receive time (see module docstring)."""
    seconds = time.time() if received_s is None else float(received_s)
    if not math.isfinite(seconds) or seconds < 0.0:
        raise ValueError("fallback capture time must be finite Unix-epoch seconds")
    stamp = round(seconds * 1_000_000_000)
    return (stamp, stamp, stamp)


def _mapping_value(mapping: Mapping[str, object], key: str, *, field: str) -> object:
    try:
        return mapping[key]
    except KeyError as error:
        raise ValueError(f"missing {field}[{key!r}]") from error


def _joint_positions(observation: Observation) -> npt.NDArray[np.float64]:
    raw = _mapping_value(observation.state, "joint_pos", field="state")
    try:
        raw_values = np.asarray(raw, dtype=object).reshape(-1)
    except Exception as error:
        raise ValueError("joint_pos must contain exactly 14 finite values") from error
    if any(isinstance(value, (bool, np.bool_)) for value in raw_values):
        raise ValueError("joint_pos must contain exactly 14 finite values")
    try:
        joints = np.asarray(raw, dtype=np.float64).reshape(-1)
    except Exception as error:
        raise ValueError("joint_pos must contain exactly 14 finite values") from error
    if joints.shape != (14,) or not np.isfinite(joints).all():
        raise ValueError("joint_pos must contain exactly 14 finite values")
    return joints


def to_dreamzero_yam(observation: Observation) -> DreamZeroYamObservation:
    """Map named Inspect cameras, times, and packed YAM state without synthesis."""
    converted, _timing = to_dreamzero_yam_with_timing(observation)
    return converted


def to_dreamzero_yam_with_timing(
    observation: Observation,
    *,
    received_s: float | None = None,
) -> tuple[DreamZeroYamObservation, CaptureTiming]:
    """Convert, and report which capture-time source the conversion used.

    ``received_s`` is the Unix-epoch time the caller received the observation;
    it is used only when the embodiment supplied no ``image_times`` at all.
    """
    joints = _joint_positions(observation)
    frames = tuple(
        _mapping_value(observation.images, name, field="images") for name in _CAMERA_NAMES
    )
    image_times = getattr(observation, "image_times", None)
    timing: CaptureTiming
    if not image_times:
        times = _fallback_capture_times_ns(received_s)
        timing = OBSERVATION_FALLBACK
    else:
        times = _capture_times_ns(observation)
        timing = PER_CAMERA
    converted = dreamscale.dreamzero_yam.observe(
        top_frame=frames[0],
        left_frame=frames[1],
        right_frame=frames[2],
        camera_capture_times_ns=times,
        left_joint_positions=joints[0:6],
        left_gripper=float(joints[6]),
        right_joint_positions=joints[7:13],
        right_gripper=float(joints[13]),
    )
    return converted, timing
