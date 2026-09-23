"""Offline Inspect policy descriptor for Dreamscale-hosted DreamZero-YAM."""

from __future__ import annotations

import atexit
import math
import statistics
import threading
import time
import warnings
from collections.abc import Callable
from numbers import Integral, Real
from typing import Any, ClassVar, Literal

import dreamscale as _dreamscale  # type: ignore[import-untyped]
import numpy as np
from dreamscale import RegionPreference, RunStrategy
from inspect_robots.policy import PolicyBase, PolicyConfig, PolicyInfo
from inspect_robots.rollout import TrialRecord
from inspect_robots.scene import Scene
from inspect_robots.spaces import (
    ActionSemantics,
    Box,
    CameraSpec,
    ObservationSpace,
    StateField,
    StateSpec,
)
from inspect_robots.types import Action, ActionChunk, Observation

from inspect_robots_dreamscale.dreamzero_yam import to_dreamzero_yam
from inspect_robots_dreamscale.telemetry import (
    TrialContext,
    runtime_identity,
    telemetry_row,
    write_trial_sidecar,
)

# Kept module-visible so discovery tests can prove construction does not call connect().
dreamscale: Any = _dreamscale

# DreamZero YAM native control contract: 30 Hz, 24 consecutive actions (0.8 s).
# Stated explicitly rather than read back from the SDK runtime contract on purpose.
# The pinned SDK now agrees, but a pin bump is a version change and this is a
# rate the qualification host must not be able to drift on silently.
YAM_CONTROL_HZ = 30.0
YAM_ACTION_HORIZON = 24

# DreamZero-YAM currently has one qualified end-to-end timebase. The SDK can
# represent other rates, but observation production, temporal admission,
# inference, and action execution have not yet been qualified as one dynamic
# contract. Fail offline rather than opening a paid session with a misleading
# partial override.

# Seconds a closed session may be held warm for the next run to reclaim. The
# control plane validates 0..3600 and rejects anything outside it at session
# create; this bound is restated so an unusable value fails during
# construction, before a cold start has been paid for.
#
# Zero -- the default -- means close really closes. Any positive value makes
# `close()` park instead of terminate, and **parked time bills at the full
# rate**, because the GPU stays reserved for you. That trade only pays off
# across an iteration loop of short, closely-spaced runs.
MAX_KEEP_WARM_S = 3600

# Fraction by which the measured step rate may differ from the commanded one
# before `act` says so. Wide on purpose: this is meant to catch an embodiment
# running at a different rate entirely, not ordinary jitter.
_RATE_WARN_TOLERANCE = 0.25
_RATE_WARN_MIN_SAMPLES = 20

YAM_DIM_LABELS = (
    "left_j0",
    "left_j1",
    "left_j2",
    "left_j3",
    "left_j4",
    "left_j5",
    "left_gripper",
    "right_j0",
    "right_j1",
    "right_j2",
    "right_j3",
    "right_j4",
    "right_j5",
    "right_gripper",
)


def _resolve_control_hz(requested: float | None) -> float:
    """Resolve the commanded rate, defaulting to the checkpoint's native one."""

    if requested is None:
        return YAM_CONTROL_HZ
    valid = (
        not isinstance(requested, bool)
        and isinstance(requested, Real)
        and math.isfinite(float(requested))
        and float(requested) == YAM_CONTROL_HZ
    )
    if not valid:
        raise ValueError(
            "DreamZero-YAM requires exactly 30 Hz; dynamic rates are not "
            "supported end to end"
        )
    return YAM_CONTROL_HZ


def _resolve_keep_warm(requested: int | None) -> int:
    """Resolve the post-close warm-hold window, in whole seconds."""

    if requested is None:
        return 0
    if isinstance(requested, bool) or not isinstance(requested, Real):
        raise ValueError("keep_warm_s must be a whole number of seconds")
    if not math.isfinite(float(requested)) or not float(requested).is_integer():
        raise ValueError("keep_warm_s must be a whole number of seconds")
    resolved = int(requested)
    if not 0 <= resolved <= MAX_KEEP_WARM_S:
        raise ValueError(
            f"keep_warm_s {resolved} is out of range; 0 disables the warm hold "
            f"and {MAX_KEEP_WARM_S} is the maximum. Held time is billed at the "
            f"full rate, so this is a cost you choose, not a free cache."
        )
    return resolved


class DreamscalePolicy(PolicyBase):
    """Describe DreamZero-YAM without reading config or opening a session."""

    # The registry name, and the prefix of every metadata key and artifact path
    # this policy writes.
    brand: ClassVar[str] = "dreamscale"

    region: RegionPreference
    sampling: Literal["upstream_eval", "async_8", "async_latest"]
    control_hz: float
    keep_warm_s: int
    startup_timeout_s: float
    timeout_s: float
    _episode_active: bool
    _closed: bool
    _rate_warned: bool
    _atexit_handler: Callable[[], None]

    def __init__(
        self,
        *,
        model: str = "dreamzero-yam",
        region: RegionPreference = "nearest",
        sampling: Literal["upstream_eval", "async_8", "async_latest"] = "async_latest",
        control_hz: float | None = None,
        keep_warm_s: int | None = None,
        startup_timeout_s: float = 1800.0,
        timeout_s: float = 60.0,
    ) -> None:
        if model != "dreamzero-yam":
            raise ValueError("only dreamzero-yam is supported")
        if sampling not in {"upstream_eval", "async_8", "async_latest"}:
            raise ValueError(
                "sampling must be upstream_eval, async_8, or async_latest"
            )
        if (
            isinstance(startup_timeout_s, bool)
            or not isinstance(startup_timeout_s, Real)
            or not math.isfinite(startup_timeout_s)
            or startup_timeout_s <= 0
        ):
            raise ValueError("startup_timeout_s must be a finite positive number")
        self.model = model
        self.region = region
        self.sampling = sampling
        self.control_hz = _resolve_control_hz(control_hz)
        self.keep_warm_s = _resolve_keep_warm(keep_warm_s)
        self.startup_timeout_s = float(startup_timeout_s)
        self.timeout_s = timeout_s
        self._remote: Any | None = None
        self._session_id: str | None = None
        self._episode_active = False
        self._closed = False
        self._trial_context: TrialContext | None = None
        self._trial_log_dir: str | None = None
        self._telemetry_rows: list[dict[str, object]] = []
        self._last_act_ns: int | None = None
        self._step_intervals_ms: list[float] = []
        self._rate_warned = False
        self.info = PolicyInfo(
            name=self.brand,
            action_space=Box(
                shape=(14,),
                semantics=ActionSemantics(
                    control_mode="joint_pos",
                    rotation_repr="none",
                    gripper="continuous",
                    frame="base",
                    dim_labels=YAM_DIM_LABELS,
                ),
            ),
            observation_space=ObservationSpace(
                cameras=(
                    CameraSpec("top_cam", 360, 640, 3),
                    CameraSpec("left_cam", 360, 640, 3),
                    CameraSpec("right_cam", 360, 640, 3),
                ),
                state=StateSpec(fields=(StateField("joint_pos", (14,), unit=""),)),
            ),
            control_hz=self.control_hz,
        )
        self.config = PolicyConfig(action_horizon=YAM_ACTION_HORIZON, replan_interval=1)
        self._atexit_handler = self._atexit_close
        atexit.register(self._atexit_handler)

    def _ensure_connected(self) -> Any:
        if self._closed:
            raise RuntimeError(f"{type(self).__name__} is closed")
        if self._remote is None:
            self._remote = dreamscale.connect(
                "dreamzero-yam",
                region=self.region,
                on_progress=None,
                startup_timeout=self.startup_timeout_s,
                # Nothing server-side takes a rate: this tells the SDK's replan
                # scheduler how fast the caller intends to execute a chunk, so
                # its latency-in-steps arithmetic matches reality.
                control_hz=int(self.control_hz),
                # Non-zero turns the SDK's close into a detach, so the session
                # parks with the model resident and the next run reclaims it
                # instead of paying another cold start. Parked time is billed.
                keep_warm=self.keep_warm_s,
            )
            self._session_id = str(self._remote.session_id)
        return self._remote

    @property
    def session_id(self) -> str | None:
        """The owned Dreamscale session identity, retained after synchronous close."""
        return self._session_id

    def prepare(self) -> None:
        """Start or reclaim compute without beginning an episode or inference.

        Callers that must acquire time-sensitive observations can prepare the
        potentially slow remote session first, then capture immediately before
        sending an observation. Repeated calls reuse the same connection.
        """
        self._ensure_connected()

    def _end_episode(self) -> None:
        if not self._episode_active:
            return
        assert self._remote is not None
        try:
            self._remote.end_episode()
        finally:
            self._episode_active = False

    def reset(self, scene: Scene) -> None:
        """Start an isolated logical episode on the reusable connection."""
        remote = self._ensure_connected()
        self._end_episode()
        self._telemetry_rows.clear()
        remote.begin_episode(
            instruction=scene.instruction,
            strategy=RunStrategy(dreamzero_sampling=self.sampling),
        )
        self._episode_active = True
        self._last_act_ns = None

    def predict_model_action(
        self,
        observation: Observation,
        *,
        instruction: str,
    ) -> Action:
        """Return one blocking model action without starting an execution episode.

        This is the non-commanding integration-check path. Unlike ``act()``, it
        waits for a raw model chunk, so ``async_latest`` cannot turn the first
        call into an expected hold. The connection is retained for a subsequent
        ``reset()`` and live Inspect episode.
        """
        if self._episode_active:
            raise RuntimeError(
                "predict_model_action() must run before reset() starts an episode"
            )
        remote = self._ensure_connected()
        result = remote.predict(
            to_dreamzero_yam(observation),
            instruction=instruction,
            timeout_s=self.timeout_s,
        )
        if not result.actions:
            raise RuntimeError("Dreamscale returned an empty model action chunk")
        action_indices = getattr(result, "action_indices", None)
        action_index = int(action_indices[0]) if action_indices else 0
        join_key = f"predict:{result.chunk_id}:{action_index}"
        return Action(
            data=np.asarray(result.actions[0], dtype=np.float64),
            meta={
                f"{self.brand}_action_source": "model",
                f"{self.brand}_chunk_id": result.chunk_id,
                f"{self.brand}_join_key": join_key,
                f"{self.brand}_observation_id": result.observation_id,
                f"{self.brand}_step": action_index,
            },
        )

    def act(self, observation: Observation) -> ActionChunk:
        """Advance the externally clocked Dreamscale episode by one Inspect step."""
        env_step = observation.extra.get("env_step")
        if isinstance(env_step, bool) or not isinstance(env_step, Integral) or env_step < 0:
            raise ValueError("extra['env_step'] must be a nonnegative integer")
        remote = self._remote
        if remote is None or not self._episode_active:
            raise RuntimeError("reset() must start an episode before act()")
        step_interval_ms = self._record_step_interval()
        started = time.perf_counter()
        result = remote.step(
            to_dreamzero_yam(observation),
            action_index=int(env_step),
            timeout_s=self.timeout_s,
        )
        wall_s = time.perf_counter() - started
        if self._trial_context is not None:
            runtime = runtime_identity(remote)
            runtime["sampling"] = self.sampling
            runtime["commanded_control_hz"] = self.control_hz
            runtime["capture_time_source"] = "embodiment_unix_epoch_seconds"
            # Recorded because a non-zero hold keeps billing after the run ends,
            # so a surprising invoice should be explicable from the sidecar.
            runtime["keep_warm_s"] = self.keep_warm_s
            self._telemetry_rows.append(
                telemetry_row(
                    result,
                    self._trial_context,
                    runtime,
                    step_interval_ms=step_interval_ms,
                )
            )
        join_key = f"{result.cache_generation}:{result.action_index}"
        action_meta = {
            f"{self.brand}_action_source": "hold" if result.stalled else "model",
            f"{self.brand}_cache_generation": result.cache_generation,
            f"{self.brand}_chunk_id": result.source_chunk_id,
            f"{self.brand}_join_key": join_key,
            f"{self.brand}_observation_id": result.observation_id,
            f"{self.brand}_step": result.action_index,
        }
        return ActionChunk(
            actions=[Action(data=np.asarray(result.action, dtype=np.float64), meta=action_meta)],
            control_hz=self.control_hz,
            inference_latency_s=wall_s,
            meta={f"{self.brand}_join_key": join_key},
        )

    def _record_step_interval(self) -> float | None:
        """Measure the wall-clock gap since the previous act(), in milliseconds."""
        now_ns = time.monotonic_ns()
        previous = self._last_act_ns
        self._last_act_ns = now_ns
        if previous is None:
            return None
        interval_ms = (now_ns - previous) / 1e6
        self._step_intervals_ms.append(interval_ms)
        self._warn_if_rate_diverges()
        return interval_ms

    def observed_control_hz(self) -> float | None:
        """The rate this episode is actually stepping at, or None if too early."""
        if len(self._step_intervals_ms) < _RATE_WARN_MIN_SAMPLES:
            return None
        median_ms = statistics.median(self._step_intervals_ms)
        if median_ms <= 0:
            return None
        return 1000.0 / median_ms

    def _warn_if_rate_diverges(self) -> None:
        """Say so, once, when the loop is not running at the commanded rate.

        Nothing enforces a control rate in this path. Inspect's rollout adds no
        wall-clock pacing of its own, so the real rate is however fast the
        embodiment returns, and `control_hz` is a declaration the SDK plans
        against rather than a rate anything imposes. A silent disagreement makes
        every replan decision wrong while the run still looks healthy. This
        cannot correct it, but it refuses to let it pass unremarked.
        """
        if self._rate_warned:
            return
        observed = self.observed_control_hz()
        if observed is None:
            return
        if abs(observed - self.control_hz) <= _RATE_WARN_TOLERANCE * self.control_hz:
            return
        self._rate_warned = True
        warnings.warn(
            f"stepping at about {observed:.1f} Hz but control_hz commands "
            f"{self.control_hz:g} Hz. Nothing enforces the rate here, so the "
            f"measured one is what the robot is doing and the commanded one is "
            f"what the action scheduler is planning against. Pace the "
            f"embodiment at the required {self.control_hz:g} Hz.",
            RuntimeWarning,
            stacklevel=3,
        )

    def on_trial_start(self, scene_id: str, epoch: int, log_dir: str, run_id: str) -> None:
        """Capture immutable artifact identity before Inspect resets the policy."""
        self._trial_context = TrialContext(run_id=run_id, scene_id=scene_id, epoch=epoch)
        self._trial_log_dir = log_dir
        self._telemetry_rows.clear()
        self._step_intervals_ms.clear()
        self._last_act_ns = None

    def on_trial_end(self, record: TrialRecord, log_dir: str, run_id: str) -> None:
        """Persist partial diagnostics even when logical episode cleanup fails."""
        del log_dir, run_id
        cleanup_error: BaseException | None = None
        try:
            self._end_episode()
        except BaseException as error:
            cleanup_error = error
        try:
            if (
                self._telemetry_rows
                and self._trial_context is not None
                and self._trial_log_dir is not None
            ):
                pointer = write_trial_sidecar(
                    self._telemetry_rows,
                    log_dir=self._trial_log_dir,
                    run_id=self._trial_context.run_id,
                    scene_id=self._trial_context.scene_id,
                    epoch=self._trial_context.epoch,
                    prefix=self.brand,
                )
                record.metadata[f"{self.brand}_telemetry"] = pointer
        finally:
            if cleanup_error is not None:
                raise cleanup_error

    def _close_suppressing_errors(self) -> None:
        try:
            self.close()
        except BaseException:
            pass

    def _atexit_close(self) -> None:
        thread = threading.Thread(target=self._close_suppressing_errors, daemon=True)
        thread.start()
        thread.join(timeout=5.0)

    def close(self) -> None:
        """End the episode and release the session once.

        With `keep_warm_s = 0` this terminates the session and billing stops.
        With a positive hold the SDK detaches instead, so the session parks with
        the model resident and **keeps billing** until it is reclaimed or the
        window expires. That is the trade: you are paying to skip the next cold
        start.
        """
        if self._closed:
            return
        self._closed = True
        atexit.unregister(self._atexit_handler)
        remote = self._remote
        if remote is None:
            return
        try:
            self._end_episode()
        finally:
            try:
                remote.close()
            finally:
                self._remote = None


def dreamscale_policy(**kwargs: Any) -> DreamscalePolicy:
    """Create the registry-discoverable Dreamscale Inspect policy."""
    return DreamscalePolicy(**kwargs)
