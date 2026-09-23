import atexit
import json
import threading
import time
import warnings
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from dreamscale import PolicyStepResult
from dreamscale.policy.config import ResolvedOptimizationConfig
from dreamscale.policy.runtime.contract import RuntimeContract
from inspect_robots.policy import Policy, PolicyConfig
from inspect_robots.rollout import TrialRecord
from inspect_robots.scene import Scene
from inspect_robots.types import Observation

from inspect_robots_dreamscale import (
    DreamscalePolicy,
    DropbearPolicy,
    dreamscale_policy,
    dropbear_policy,
)


class FakeRemotePolicy:
    model = "dreamzero-yam"
    session_id = "session-123"
    region = "us-west-2"
    _target_key = "a3-ga"
    transport_mode = "quic"
    fallback_reason = None
    runtime_contract = RuntimeContract(
        action_hz=30.0,
        chunk_size=24,
        action_dim=14,
        action_space="raw_absolute_joint",
        hold_behavior="current_position",
    )
    resolved_optimization_config = ResolvedOptimizationConfig()

    def __init__(
        self,
        *,
        step_result: PolicyStepResult | None = None,
        predict_result: object | None = None,
        begin_error: Exception | None = None,
        end_error: Exception | None = None,
    ) -> None:
        self.begin_calls: list[tuple[str, str]] = []
        self.end_calls = 0
        self.close_calls = 0
        self.step_result = step_result
        self.predict_result = predict_result
        self.predict_calls: list[tuple[str, float]] = []
        self.step_indices: list[int] = []
        self.step_timeouts: list[float] = []
        self.begin_error = begin_error
        self.end_error = end_error

    def begin_episode(self, *, instruction: str, strategy: Any) -> None:
        self.begin_calls.append((instruction, strategy.dreamzero_sampling))
        if self.begin_error is not None:
            raise self.begin_error

    def end_episode(self) -> None:
        self.end_calls += 1
        if self.end_error is not None:
            raise self.end_error

    def close(self) -> None:
        self.close_calls += 1

    def predict(self, observation: object, *, instruction: str, timeout_s: float) -> object:
        del observation
        self.predict_calls.append((instruction, timeout_s))
        assert self.predict_result is not None
        return self.predict_result

    def step(self, observation: object, *, action_index: int, timeout_s: float) -> PolicyStepResult:
        del observation
        self.step_indices.append(action_index)
        self.step_timeouts.append(timeout_s)
        assert self.step_result is not None
        return self.step_result


def inspect_observation(*, env_step: object = 8) -> Observation:
    frame = np.zeros((2, 3, 3), dtype=np.uint8)
    return Observation(
        images={"top_cam": frame, "left_cam": frame, "right_cam": frame},
        state={"joint_pos": np.arange(14, dtype=np.float64)},
        instruction="spell NEURIPS",
        image_times={
            "top_cam": time.time(),
            "left_cam": time.time() + 0.001,
            "right_cam": time.time() + 0.002,
        },
        extra={"env_step": env_step},
    )


def step_result(*, stalled: bool = False) -> PolicyStepResult:
    return PolicyStepResult(
        action=tuple(float(i) for i in range(14)),
        action_index=8,
        observation_id=11,
        stalled=stalled,
        actions_remaining=15,
        source_chunk_id=4,
        source_observation_id=9,
        source_control_tick=7,
        source_capture_to_execution_ms=81.25,
        replan_epoch=1,
        cache_generation=2,
    )


def test_factory_constructs_only_dreamzero_yam_without_connecting(monkeypatch) -> None:
    """Catch a discovery path that opens a Dreamscale connection."""
    monkeypatch.setattr(
        "inspect_robots_dreamscale.policy.dreamscale.connect",
        lambda *_args, **_kwargs: pytest.fail("constructor touched the network"),
    )

    policy = dreamscale_policy(model="dreamzero-yam")

    assert isinstance(policy, Policy)
    assert policy.info.name == "dreamscale"
    assert policy.info.action_space.shape == (14,)
    assert policy.info.control_hz == 30.0
    assert policy.config == PolicyConfig(action_horizon=24, replan_interval=1)
    assert policy.sampling == "async_latest"


def test_factory_rejects_every_other_model_offline() -> None:
    """Catch a factory that accepts a Dreamscale model other than DreamZero-YAM."""
    with pytest.raises(ValueError, match="only dreamzero-yam is supported"):
        dreamscale_policy(model="dreamzero-droid")


def test_factory_rejects_unknown_sampling_mode_offline() -> None:
    """Catch exposing an unsupported scheduler through the private adapter."""
    with pytest.raises(
        ValueError,
        match="sampling must be upstream_eval, async_8, or async_latest",
    ):
        dreamscale_policy(model="dreamzero-yam", sampling="custom")


def test_factory_accepts_explicit_async_latest_without_connecting(monkeypatch) -> None:
    monkeypatch.setattr(
        "inspect_robots_dreamscale.policy.dreamscale.connect",
        lambda *_args, **_kwargs: pytest.fail("constructor touched the network"),
    )

    policy = dreamscale_policy(model="dreamzero-yam", sampling="async_latest")

    assert policy.sampling == "async_latest"


def test_reset_passes_default_startup_timeout_to_dreamscale_connect(monkeypatch) -> None:
    """Catch restoring the SDK's shorter default startup deadline for DreamZero-YAM."""
    remote = FakeRemotePolicy()
    startup_timeouts: list[float] = []

    def connect(
        _model: str,
        *,
        region: str,
        on_progress: object,
        startup_timeout: float,
        control_hz: int,
        keep_warm: int,
    ) -> FakeRemotePolicy:
        del region, on_progress, control_hz, keep_warm
        startup_timeouts.append(startup_timeout)
        return remote

    monkeypatch.setattr("inspect_robots_dreamscale.policy.dreamscale.connect", connect)
    policy = dreamscale_policy(model="dreamzero-yam")

    policy.reset(Scene(id="spell", instruction="spell NEURIPS"))

    assert startup_timeouts == [1800.0]


def test_prepare_connects_once_without_starting_an_episode_or_inference(monkeypatch) -> None:
    remote = FakeRemotePolicy()
    connect_calls = 0

    def connect(*_args: object, **_kwargs: object) -> FakeRemotePolicy:
        nonlocal connect_calls
        connect_calls += 1
        return remote

    monkeypatch.setattr("inspect_robots_dreamscale.policy.dreamscale.connect", connect)
    policy = dreamscale_policy(model="dreamzero-yam")

    policy.prepare()
    policy.prepare()

    assert connect_calls == 1
    assert policy.session_id == "session-123"
    assert remote.begin_calls == []
    assert remote.predict_calls == []


def test_custom_startup_timeout_does_not_change_per_step_timeout(monkeypatch) -> None:
    """Catch startup and action-step deadlines being conflated at the SDK boundary."""
    remote = FakeRemotePolicy(step_result=step_result())
    startup_timeouts: list[float] = []

    def connect(
        _model: str,
        *,
        region: str,
        on_progress: object,
        startup_timeout: float,
        control_hz: int,
        keep_warm: int,
    ) -> FakeRemotePolicy:
        del region, on_progress, control_hz, keep_warm
        startup_timeouts.append(startup_timeout)
        return remote

    monkeypatch.setattr("inspect_robots_dreamscale.policy.dreamscale.connect", connect)
    policy = dreamscale_policy(
        model="dreamzero-yam",
        startup_timeout_s=2400.0,
        timeout_s=17.0,
    )

    policy.reset(Scene(id="spell", instruction="spell NEURIPS"))
    policy.act(inspect_observation())

    assert startup_timeouts == [2400.0]
    assert remote.step_timeouts == [17.0]


def test_blocking_model_prediction_reuses_connection_without_starting_episode(
    monkeypatch,
) -> None:
    result = SimpleNamespace(
        actions=(tuple(float(i) for i in range(14)),),
        action_indices=(0,),
        chunk_id=7,
        observation_id=11,
    )
    remote = FakeRemotePolicy(predict_result=result)
    connect_calls = 0

    def connect(*_args, **_kwargs) -> FakeRemotePolicy:
        nonlocal connect_calls
        connect_calls += 1
        return remote

    monkeypatch.setattr("inspect_robots_dreamscale.policy.dreamscale.connect", connect)
    policy = dreamscale_policy(model="dreamzero-yam", timeout_s=17.0)

    action = policy.predict_model_action(
        inspect_observation(env_step=0),
        instruction="spell NEURIPS",
    )
    policy.reset(Scene(id="live", instruction="spell NEURIPS"))

    assert connect_calls == 1
    assert remote.predict_calls == [("spell NEURIPS", 17.0)]
    assert remote.begin_calls == [("spell NEURIPS", "async_latest")]
    assert np.array_equal(action.data, np.arange(14, dtype=np.float64))
    assert action.meta == {
        "dreamscale_action_source": "model",
        "dreamscale_chunk_id": 7,
        "dreamscale_join_key": "predict:7:0",
        "dreamscale_observation_id": 11,
        "dreamscale_step": 0,
    }


@pytest.mark.parametrize(
    "startup_timeout_s",
    [True, False, None, "1800", 0, -1, float("nan"), float("inf"), float("-inf")],
)
def test_factory_rejects_invalid_startup_timeout_before_connect(
    monkeypatch, startup_timeout_s
) -> None:
    """Catch malformed startup budgets reaching config/session side effects."""
    monkeypatch.setattr(
        "inspect_robots_dreamscale.policy.dreamscale.connect",
        lambda *_args, **_kwargs: pytest.fail("invalid timeout opened a connection"),
    )

    with pytest.raises(ValueError, match="startup_timeout_s must be a finite positive number"):
        dreamscale_policy(model="dreamzero-yam", startup_timeout_s=startup_timeout_s)


def test_reset_reuses_connection_but_isolates_same_instruction_episodes(monkeypatch) -> None:
    """Catch connection churn or reused episode state between identical trials."""
    remote = FakeRemotePolicy()
    connects: list[tuple[str, str, object]] = []

    def connect(
        model: str,
        *,
        region: str,
        on_progress: object,
        startup_timeout: float,
        control_hz: int,
        keep_warm: int,
    ) -> FakeRemotePolicy:
        del startup_timeout, control_hz, keep_warm
        connects.append((model, region, on_progress))
        return remote

    monkeypatch.setattr("inspect_robots_dreamscale.policy.dreamscale.connect", connect)
    policy = dreamscale_policy(model="dreamzero-yam", sampling="upstream_eval")
    scene = Scene(id="spell", instruction="spell NEURIPS")

    policy.reset(scene)
    policy.reset(scene)
    policy.close()

    assert connects == [("dreamzero-yam", "nearest", None)]
    assert remote.begin_calls == [
        ("spell NEURIPS", "upstream_eval"),
        ("spell NEURIPS", "upstream_eval"),
    ]
    assert remote.end_calls == 2
    assert remote.close_calls == 1


def test_close_before_reset_is_offline_and_idempotent(monkeypatch) -> None:
    """Catch close accidentally connecting or releasing an owned remote twice."""
    monkeypatch.setattr(
        "inspect_robots_dreamscale.policy.dreamscale.connect",
        lambda *_args, **_kwargs: pytest.fail("close touched the network"),
    )
    policy = dreamscale_policy(model="dreamzero-yam")

    policy.close()
    policy.close()


def test_session_identity_is_read_only_and_survives_synchronous_close(monkeypatch) -> None:
    """Let the composition runner verify cleanup after the remote is released."""
    remote = FakeRemotePolicy()
    monkeypatch.setattr(
        "inspect_robots_dreamscale.policy.dreamscale.connect",
        lambda *_args, **_kwargs: remote,
    )
    policy = dreamscale_policy(model="dreamzero-yam")

    assert policy.session_id is None
    policy.reset(Scene(id="spell", instruction="spell NEURIPS"))
    assert policy.session_id == "session-123"

    with pytest.raises(AttributeError):
        policy.session_id = "different"

    policy.close()
    policy.close()

    assert policy.session_id == "session-123"
    assert remote.close_calls == 1


@pytest.mark.parametrize(("stalled", "source"), [(False, "model"), (True, "hold")])
def test_act_uses_env_step_and_returns_one_joinable_action(monkeypatch, stalled, source) -> None:
    """Catch a shifted control index, leaked chunk, or wrong action-source marker."""
    remote = FakeRemotePolicy(step_result=step_result(stalled=stalled))
    monkeypatch.setattr(
        "inspect_robots_dreamscale.policy.dreamscale.connect",
        lambda *_args, **_kwargs: remote,
    )
    policy = dreamscale_policy(model="dreamzero-yam", sampling="upstream_eval")
    policy.reset(Scene(id="spell", instruction="spell NEURIPS"))

    chunk = policy.act(inspect_observation(env_step=8))

    assert remote.step_indices == [8]
    assert len(chunk.actions) == 1
    assert chunk.actions[0].data.tolist() == [float(i) for i in range(14)]
    assert chunk.actions[0].meta == {
        "dreamscale_action_source": source,
        "dreamscale_cache_generation": 2,
        "dreamscale_chunk_id": 4,
        "dreamscale_join_key": "2:8",
        "dreamscale_observation_id": 11,
        "dreamscale_step": 8,
    }
    assert chunk.control_hz == 30.0
    assert chunk.inference_latency_s is not None
    assert chunk.inference_latency_s >= 0.0
    assert chunk.meta == {"dreamscale_join_key": "2:8"}


@pytest.mark.parametrize("env_step", [None, True, 8.0, -1, "8"])
def test_act_rejects_malformed_env_step_before_remote_step(monkeypatch, env_step) -> None:
    """Catch an invalid Inspect clock value crossing the Dreamscale boundary."""
    remote = FakeRemotePolicy(step_result=step_result())
    monkeypatch.setattr(
        "inspect_robots_dreamscale.policy.dreamscale.connect",
        lambda *_args, **_kwargs: remote,
    )
    policy = dreamscale_policy(model="dreamzero-yam")
    policy.reset(Scene(id="spell", instruction="spell NEURIPS"))
    observation = inspect_observation(env_step=env_step)
    if env_step is None:
        observation = Observation(
            images=observation.images,
            state=observation.state,
            image_times=observation.image_times,
            extra={},
        )

    with pytest.raises(ValueError, match=r"extra\['env_step'\] must be a nonnegative integer"):
        policy.act(observation)

    assert remote.step_indices == []


def test_trial_end_preserves_partial_row_and_pointer_before_reraising_cleanup_error(
    monkeypatch, tmp_path: Path
) -> None:
    """Catch cleanup failure discarding the serving evidence needed to debug it."""
    remote = FakeRemotePolicy(step_result=step_result(), end_error=RuntimeError("end failed"))
    monkeypatch.setattr(
        "inspect_robots_dreamscale.policy.dreamscale.connect",
        lambda *_args, **_kwargs: remote,
    )
    policy = dreamscale_policy(model="dreamzero-yam")
    policy.on_trial_start("spell", 2, str(tmp_path), "run-123")
    policy.reset(Scene(id="spell", instruction="spell NEURIPS"))
    policy.act(inspect_observation())
    record = TrialRecord(scene_id="spell", epoch=2, seed=7, metadata={"existing": "kept"})

    with pytest.raises(RuntimeError, match="end failed"):
        policy.on_trial_end(record, str(tmp_path), "run-123")

    assert record.metadata == {
        "existing": "kept",
        "dreamscale_telemetry": "dreamscale/run-123/spell-e2.jsonl",
    }
    sidecar = tmp_path / record.metadata["dreamscale_telemetry"]
    rows = [json.loads(line) for line in sidecar.read_text().splitlines()]
    assert len(rows) == 1
    assert rows[0]["join_key"] == "2:8"
    assert rows[0]["run_id"] == "run-123"


def test_failed_pre_episode_reset_writes_no_empty_sidecar_or_pointer(
    monkeypatch, tmp_path: Path
) -> None:
    """Catch a failed reset being misrepresented as a delivered telemetry artifact."""
    remote = FakeRemotePolicy(begin_error=RuntimeError("begin failed"))
    monkeypatch.setattr(
        "inspect_robots_dreamscale.policy.dreamscale.connect",
        lambda *_args, **_kwargs: remote,
    )
    policy = dreamscale_policy(model="dreamzero-yam")
    policy.on_trial_start("spell", 0, str(tmp_path), "run-123")
    with pytest.raises(RuntimeError, match="begin failed"):
        policy.reset(Scene(id="spell", instruction="spell NEURIPS"))
    record = TrialRecord(scene_id="spell", epoch=0, seed=7)

    policy.on_trial_end(record, str(tmp_path), "run-123")

    assert "dreamscale_telemetry" not in record.metadata
    assert not (tmp_path / "dreamscale").exists()


def test_telemetry_records_runtime_transport_state_at_action_time(
    monkeypatch, tmp_path: Path
) -> None:
    """Catch a post-connect transport fallback being hidden by stale runtime identity."""
    remote = FakeRemotePolicy(step_result=step_result())
    monkeypatch.setattr(
        "inspect_robots_dreamscale.policy.dreamscale.connect",
        lambda *_args, **_kwargs: remote,
    )
    policy = dreamscale_policy(model="dreamzero-yam")
    policy.on_trial_start("spell", 0, str(tmp_path), "run-123")
    policy.reset(Scene(id="spell", instruction="spell NEURIPS"))
    remote.transport_mode = "relay"
    remote.fallback_reason = "quic_result_timeout"
    policy.act(inspect_observation())
    record = TrialRecord(scene_id="spell", epoch=0, seed=7)

    policy.on_trial_end(record, str(tmp_path), "run-123")

    row = json.loads((tmp_path / record.metadata["dreamscale_telemetry"]).read_text())
    assert row["runtime"]["transport_mode"] == "relay"
    assert row["runtime"]["fallback_reason"] == "quic_result_timeout"
    assert row["runtime"]["sampling"] == "async_latest"
    assert row["runtime"]["capture_time_source"] == "embodiment_unix_epoch_seconds"
    assert row["runtime"]["commanded_control_hz"] == 30.0
    assert row["runtime"]["session_id"] == "session-123"
    assert row["action_source"] == "model"


def test_explicit_close_unregisters_single_atexit_handler(monkeypatch) -> None:
    """Catch duplicate exit cleanup or an explicit close leaving fallback registered."""
    registered: list[object] = []
    unregistered: list[object] = []
    monkeypatch.setattr(atexit, "register", lambda handler: registered.append(handler) or handler)
    monkeypatch.setattr(atexit, "unregister", unregistered.append)

    policy = dreamscale_policy(model="dreamzero-yam")
    policy.close()
    policy.close()

    assert len(registered) == 1
    assert unregistered == registered


def test_atexit_fallback_uses_bounded_daemon_thread(monkeypatch) -> None:
    """Catch process exit blocking indefinitely on remote cleanup."""
    created: list[object] = []
    joins: list[float] = []

    class FakeThread:
        def __init__(self, *, target, daemon: bool) -> None:
            self.target = target
            self.daemon = daemon
            created.append(self)

        def start(self) -> None:
            self.target()

        def join(self, *, timeout: float) -> None:
            joins.append(timeout)

    monkeypatch.setattr(threading, "Thread", FakeThread)
    policy = dreamscale_policy(model="dreamzero-yam")

    policy._atexit_close()

    assert len(created) == 1
    assert created[0].daemon is True
    assert joins == [5.0]


def _connect_recording(remote: "FakeRemotePolicy", sink: list[int], warm_sink=None):
    """A connect stub recording the rate and warm hold the adapter asked for."""
    warm_sink = [] if warm_sink is None else warm_sink

    def connect(
        _model: str,
        *,
        region: str,
        on_progress: object,
        startup_timeout: float,
        control_hz: int,
        keep_warm: int,
    ) -> FakeRemotePolicy:
        del region, on_progress, startup_timeout
        sink.append(control_hz)
        warm_sink.append(keep_warm)
        return remote

    return connect


def test_control_hz_defaults_to_the_native_rate(monkeypatch) -> None:
    """Catch a default that stops matching the checkpoint's own 30 Hz."""
    remote = FakeRemotePolicy(step_result=step_result())
    rates: list[int] = []
    monkeypatch.setattr(
        "inspect_robots_dreamscale.policy.dreamscale.connect",
        _connect_recording(remote, rates),
    )
    policy = dreamscale_policy(model="dreamzero-yam")

    policy.reset(Scene(id="spell", instruction="spell NEURIPS"))
    chunk = policy.act(inspect_observation())

    assert policy.control_hz == 30.0
    assert policy.info.control_hz == 30.0
    assert chunk.control_hz == 30.0
    assert rates == [30]


@pytest.mark.parametrize(
    "control_hz",
    [4, 5, 15, 29, 31, 0, -15, 15.5, True, False, "30", float("nan")],
)
def test_unusable_control_hz_fails_before_a_session_is_opened(
    monkeypatch, control_hz: object
) -> None:
    """Catch an unusable rate that only surfaces after paying for a cold start."""
    monkeypatch.setattr(
        "inspect_robots_dreamscale.policy.dreamscale.connect",
        lambda *_args, **_kwargs: pytest.fail("invalid control_hz opened a connection"),
    )
    with pytest.raises(ValueError, match="requires exactly 30 Hz"):
        dreamscale_policy(model="dreamzero-yam", control_hz=control_hz)


def test_step_interval_is_recorded_and_absent_on_the_first_step(monkeypatch) -> None:
    """Catch losing the only measurement of the rate the loop actually ran at."""
    remote = FakeRemotePolicy(step_result=step_result())
    monkeypatch.setattr(
        "inspect_robots_dreamscale.policy.dreamscale.connect",
        _connect_recording(remote, []),
    )
    policy = dreamscale_policy(model="dreamzero-yam")
    policy.on_trial_start("spell", 0, "logs", "run-1")
    policy.reset(Scene(id="spell", instruction="spell NEURIPS"))

    for _ in range(3):
        policy.act(inspect_observation())

    rows = policy._telemetry_rows
    assert len(rows) == 3
    # Absent rather than zero-filled: there is no previous step to measure from.
    assert "step_interval_ms" not in rows[0]
    assert all(row["step_interval_ms"] > 0 for row in rows[1:])
    assert all(row["runtime"]["commanded_control_hz"] == 30.0 for row in rows)


def test_a_loop_running_at_the_wrong_rate_is_reported(monkeypatch) -> None:
    """Catch silently planning against a rate the robot is not running at.

    Nothing enforces the commanded rate in this path, so a mismatch has to be
    observed to be known. This is the check that makes `control_hz` a claim the
    adapter can contradict rather than one it simply trusts.
    """
    remote = FakeRemotePolicy(step_result=step_result())
    monkeypatch.setattr(
        "inspect_robots_dreamscale.policy.dreamscale.connect",
        _connect_recording(remote, []),
    )
    # Command 30 Hz, then step at 10 Hz (100 ms apart) by advancing the clock.
    now_ns = [0]
    monkeypatch.setattr(
        "inspect_robots_dreamscale.policy.time.monotonic_ns", lambda: now_ns[0]
    )
    policy = dreamscale_policy(model="dreamzero-yam", control_hz=30)
    policy.reset(Scene(id="spell", instruction="spell NEURIPS"))

    with pytest.warns(RuntimeWarning, match="control_hz commands 30 Hz"):
        for _ in range(25):
            policy.act(inspect_observation())
            now_ns[0] += 100_000_000

    assert policy.observed_control_hz() == pytest.approx(10.0)


def test_a_loop_running_at_the_commanded_rate_is_silent(monkeypatch) -> None:
    """Catch a rate check noisy enough that people learn to ignore it."""
    remote = FakeRemotePolicy(step_result=step_result())
    monkeypatch.setattr(
        "inspect_robots_dreamscale.policy.dreamscale.connect",
        _connect_recording(remote, []),
    )
    now_ns = [0]
    monkeypatch.setattr(
        "inspect_robots_dreamscale.policy.time.monotonic_ns", lambda: now_ns[0]
    )
    policy = dreamscale_policy(model="dreamzero-yam", control_hz=30)
    policy.reset(Scene(id="spell", instruction="spell NEURIPS"))

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        for _ in range(25):
            policy.act(inspect_observation())
            # 30 Hz is 33.33 ms; jitter either side stays inside tolerance.
            now_ns[0] += 33_333_000

    assert policy.observed_control_hz() == pytest.approx(30.0, rel=1e-3)


def test_keep_warm_defaults_to_off(monkeypatch) -> None:
    """Catch a default that silently keeps billing after the run ends.

    A warm hold reserves the GPU and bills for it, so it must be something the
    caller asks for. Defaulting it on would charge people for iterating.
    """
    remote = FakeRemotePolicy(step_result=step_result())
    warm: list[int] = []
    monkeypatch.setattr(
        "inspect_robots_dreamscale.policy.dreamscale.connect",
        _connect_recording(remote, [], warm),
    )
    policy = dreamscale_policy(model="dreamzero-yam")

    policy.reset(Scene(id="spell", instruction="spell NEURIPS"))

    assert policy.keep_warm_s == 0
    assert warm == [0]


def test_keep_warm_reaches_connect(monkeypatch) -> None:
    """Catch the hold being accepted but never sent, so nothing parks."""
    remote = FakeRemotePolicy(step_result=step_result())
    warm: list[int] = []
    monkeypatch.setattr(
        "inspect_robots_dreamscale.policy.dreamscale.connect",
        _connect_recording(remote, [], warm),
    )
    policy = dreamscale_policy(model="dreamzero-yam", keep_warm_s=300)

    policy.reset(Scene(id="spell", instruction="spell NEURIPS"))
    policy.act(inspect_observation())

    assert policy.keep_warm_s == 300
    assert warm == [300]


@pytest.mark.parametrize("keep_warm_s", [-1, 3601, 10.5, True, False, "300", float("nan")])
def test_unusable_keep_warm_fails_before_a_session_is_opened(
    monkeypatch, keep_warm_s: object
) -> None:
    """Catch an unusable hold that only surfaces after paying a cold start."""
    monkeypatch.setattr(
        "inspect_robots_dreamscale.policy.dreamscale.connect",
        lambda *_args, **_kwargs: pytest.fail("invalid keep_warm_s opened a connection"),
    )
    with pytest.raises(ValueError, match="keep_warm_s"):
        dreamscale_policy(model="dreamzero-yam", keep_warm_s=keep_warm_s)


def test_keep_warm_bounds_are_inclusive(monkeypatch) -> None:
    """Catch an off-by-one at either end of the documented range."""
    monkeypatch.setattr(
        "inspect_robots_dreamscale.policy.dreamscale.connect",
        lambda *_args, **_kwargs: pytest.fail("constructor touched the network"),
    )
    assert dreamscale_policy(model="dreamzero-yam", keep_warm_s=0).keep_warm_s == 0
    assert dreamscale_policy(model="dreamzero-yam", keep_warm_s=3600).keep_warm_s == 3600


def test_keep_warm_is_recorded_in_telemetry(monkeypatch) -> None:
    """Catch a surprising invoice that cannot be explained from the sidecar."""
    remote = FakeRemotePolicy(step_result=step_result())
    monkeypatch.setattr(
        "inspect_robots_dreamscale.policy.dreamscale.connect",
        _connect_recording(remote, []),
    )
    policy = dreamscale_policy(model="dreamzero-yam", keep_warm_s=120)
    policy.on_trial_start("spell", 0, "logs", "run-1")
    policy.reset(Scene(id="spell", instruction="spell NEURIPS"))
    policy.act(inspect_observation())

    assert policy._telemetry_rows[0]["runtime"]["keep_warm_s"] == 120


def test_both_policy_names_are_registered_entry_points() -> None:
    """Catch dropping the new name, or the pre-rename alias existing configs use."""
    from importlib.metadata import entry_points

    names = {ep.name: ep.value for ep in entry_points(group="inspect_robots.policies")}

    assert names["dreamscale"] == "inspect_robots_dreamscale:dreamscale_policy"
    assert names["dropbear"] == "inspect_robots_dreamscale:dropbear_policy"


def test_dropbear_alias_is_the_same_policy_with_pre_rename_artifact_names(
    monkeypatch, tmp_path: Path
) -> None:
    """Catch the alias silently renaming artifacts that existing tooling reads."""
    remote = FakeRemotePolicy(step_result=step_result())
    monkeypatch.setattr(
        "inspect_robots_dreamscale.policy.dreamscale.connect",
        lambda *_args, **_kwargs: remote,
    )
    policy = dropbear_policy(model="dreamzero-yam")
    assert isinstance(policy, DreamscalePolicy)
    assert isinstance(policy, DropbearPolicy)
    assert policy.info.name == "dropbear"

    policy.on_trial_start("spell", 0, str(tmp_path), "run-123")
    policy.reset(Scene(id="spell", instruction="spell NEURIPS"))
    chunk = policy.act(inspect_observation())
    record = TrialRecord(scene_id="spell", epoch=0, seed=7)
    policy.on_trial_end(record, str(tmp_path), "run-123")

    assert chunk.meta == {"dropbear_join_key": "2:8"}
    assert chunk.actions[0].meta["dropbear_action_source"] == "model"
    assert record.metadata == {"dropbear_telemetry": "dropbear/run-123/spell-e0.jsonl"}
    assert (tmp_path / "dropbear/run-123/spell-e0.jsonl").is_file()
    assert not (tmp_path / "dreamscale").exists()


def test_closed_error_names_the_class_the_caller_constructed() -> None:
    """Catch a closed-policy error naming a class the caller never used."""
    policy = dreamscale_policy(model="dreamzero-yam")
    policy.close()
    with pytest.raises(RuntimeError, match="^DreamscalePolicy is closed$"):
        policy.prepare()
