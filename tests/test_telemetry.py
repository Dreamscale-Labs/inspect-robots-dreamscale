import json
import re
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
from dreamscale import ChunkEvent, MergeEvent, PolicyStepResult
from dreamscale.policy.config import ResolvedOptimizationConfig
from dreamscale.policy.runtime.contract import RuntimeContract
from dreamscale.policy.types import PolicyTimingEvent

from inspect_robots_dreamscale.telemetry import (
    TrialContext,
    runtime_identity,
    telemetry_row,
    write_trial_sidecar,
)


def result_with_events() -> PolicyStepResult:
    timing = PolicyTimingEvent(
        observation_id=11,
        chunk_id=4,
        obs_to_action_ms=42.5,
        server_queue_ms=None,
        server_inference_ms=20.0,
        server_total_ms=25.0,
        client_overhead_ms=float("nan"),
        transport_send_ms=1.0,
        wait_for_chunk_ms=30.0,
        quic_rtt_ms=4.0,
        transport_mode="quic",
    )
    chunk = ChunkEvent(
        observation_id=11,
        chunk_id=4,
        base_action_index=8,
        accepted_action_index=8,
        actions=((100.0, 200.0), (300.0, 400.0)),
        action_indices=(8, 9),
    )
    merge = MergeEvent(
        observation_id=11,
        chunk_id=4,
        base_action_index=8,
        accepted_action_index=8,
        requested_mode="output",
        resolved_mode="output",
        new_action_weight=0.5,
        correction_steps=2,
        stale_steps=0,
        overlap_steps=8,
        resolved_steps=8,
        rebase_offset_steps=0,
        first_executable_source_offset=0,
        max_abs_position_revision=0.02,
    )
    return PolicyStepResult(
        action=tuple(float(i) for i in range(14)),
        action_index=8,
        observation_id=11,
        stalled=False,
        actions_remaining=15,
        source_chunk_id=4,
        source_observation_id=9,
        source_control_tick=7,
        source_capture_to_execution_ms=81.25,
        replan_epoch=1,
        cache_generation=2,
        timing_events=(timing, timing),
        chunk_events=(chunk, chunk),
        merge_events=(merge, merge),
    )


def context() -> TrialContext:
    return TrialContext(run_id="20260812_010203_deadbeef", scene_id="spell/NEURIPS", epoch=0)


def test_telemetry_row_is_joinable_bounded_and_strict_json() -> None:
    """Catch action/image leakage or missing join fields in serving diagnostics."""
    runtime = {
        "model": "dreamzero-yam",
        "nested": {"missing": None, "numpy_nan": np.float32("nan"), "valid": 15.0},
    }

    row = telemetry_row(result_with_events(), context(), runtime)
    encoded = json.dumps(row, allow_nan=False, sort_keys=True)

    assert row["schema_version"] == 2
    assert row["env_step"] == 8
    assert row["logical_action_index"] == 8
    assert row["join_key"] == "2:8"
    assert row["action_source"] == "model"
    assert row["source_control_tick"] == 7
    assert row["source_capture_to_execution_ms"] == 81.25
    assert len(row["timing"]) == 2
    assert len(row["chunks"]) == 2
    assert len(row["merges"]) == 2
    assert row["runtime"] == {"model": "dreamzero-yam", "nested": {"valid": 15.0}}
    assert "server_queue_ms" not in row["timing"][0]
    assert "client_overhead_ms" not in row["timing"][0]
    assert "actions" not in row["chunks"][0]
    assert row["merges"][0]["max_abs_position_revision"] == 0.02
    for forbidden in ('"action"', '"actions"', '"image"', '"camera"'):
        assert forbidden not in encoded


@dataclass(frozen=True)
class LeakyTimingEvent(PolicyTimingEvent):
    authorization: str = "Bearer secret"
    camera: bytes = b"image"
    actions: tuple[float, ...] = (1.0, 2.0)


def test_telemetry_event_projection_ignores_fields_outside_schema_v2() -> None:
    """Catch a future SDK event field silently expanding the sidecar schema."""
    original = result_with_events().timing_events[0]
    leaky = LeakyTimingEvent(**original.__dict__)
    result = replace(result_with_events(), timing_events=(leaky,))

    encoded = json.dumps(telemetry_row(result, context(), {}), allow_nan=False, sort_keys=True)

    assert "authorization" not in encoded
    assert '"camera"' not in encoded
    assert '"actions"' not in encoded
    assert "secret" not in encoded


def test_hold_telemetry_omits_false_source_provenance() -> None:
    result = replace(
        result_with_events(),
        stalled=True,
        source_chunk_id=None,
        source_observation_id=None,
        source_control_tick=None,
        source_capture_to_execution_ms=None,
        timing_events=(),
        chunk_events=(),
        merge_events=(),
    )

    row = telemetry_row(result, context(), {})

    assert row["action_source"] == "hold"
    assert "source_chunk_id" not in row
    assert "source_observation_id" not in row
    assert "source_control_tick" not in row
    assert "source_capture_to_execution_ms" not in row


@dataclass(frozen=True)
class FakeRemote:
    model: str = "dreamzero-yam"
    session_id: str = "session-123"
    region: str = "us-west-2"
    _target_key: str = "a3-ga"
    transport_mode: str = "quic"
    fallback_reason: str | None = None
    runtime_contract: RuntimeContract = RuntimeContract(
        action_hz=30.0,
        chunk_size=24,
        action_dim=14,
        action_space="raw_absolute_joint",
        hold_behavior="current_position",
    )
    resolved_optimization_config: ResolvedOptimizationConfig = ResolvedOptimizationConfig()
    api_key: str = "secret-api-key"
    authorization: str = "Bearer secret"
    session_token: str = "secret-token"
    certificate: str = "secret-cert"
    endpoint: str = "https://secret.example"
    camera: bytes = b"camera-bytes"
    actions: tuple[float, ...] = (1.0, 2.0)


def test_runtime_identity_uses_only_explicitly_whitelisted_fields(monkeypatch) -> None:
    """Catch broad remote serialization that would leak credentials or payloads."""
    package_versions = {
        "dreamscale": "0.1.0a10",
        "inspect-robots": "0.1.0",
        "inspect-robots-dreamscale": "0.1.3",
    }
    monkeypatch.setattr(
        "inspect_robots_dreamscale.telemetry.importlib.metadata.version",
        lambda package: package_versions[package],
    )

    identity = runtime_identity(FakeRemote())
    encoded = json.dumps(identity, allow_nan=False, sort_keys=True)

    assert identity["model"] == "dreamzero-yam"
    assert identity["target_key"] == "a3-ga"
    assert identity["runtime_contract"]["chunk_size"] == 24
    assert identity["resolved_optimization_config"]["backend"] == "pytorch"
    assert identity["packages"] == {
        "dreamscale": "0.1.0a10",
        "inspect_robots": "0.1.0",
        "inspect_robots_dreamscale": "0.1.3",
    }
    for forbidden in (
        "api_key",
        "authorization",
        "session_token",
        "certificate",
        "endpoint",
        "camera",
        "actions",
        "secret",
    ):
        assert forbidden not in encoded.lower()


def test_write_trial_sidecar_is_atomic_sanitized_and_does_not_mutate_rows(
    tmp_path: Path,
) -> None:
    """Catch partial files, traversal, invalid JSON, or caller-owned row mutation."""
    rows = [
        {
            "ok": 1,
            "missing": None,
            "nested": {"infinite": float("inf"), "valid": 2.0},
            "items": [None, float("nan"), "kept"],
        }
    ]

    pointer = write_trial_sidecar(
        rows,
        log_dir=tmp_path,
        run_id="20260812_010203_deadbeef",
        scene_id="spell/NEURIPS",
        epoch=0,
    )

    assert pointer == "dreamscale/20260812_010203_deadbeef/spell-NEURIPS-e0.jsonl"
    assert re.fullmatch(r"dreamscale/[A-Za-z0-9._-]+/[A-Za-z0-9._-]+", pointer)
    path = tmp_path / pointer
    assert json.loads(path.read_text()) == {
        "items": ["kept"],
        "nested": {"valid": 2.0},
        "ok": 1,
    }
    assert not list(path.parent.glob("*.tmp"))
    assert "missing" in rows[0]
    assert "infinite" in rows[0]["nested"]
    assert len(rows[0]["items"]) == 3


def test_write_trial_sidecar_sanitizes_and_caps_each_path_component(tmp_path: Path) -> None:
    """Catch directory traversal or unbounded customer identifiers in artifact paths."""
    pointer = write_trial_sidecar(
        [{"schema_version": 1}],
        log_dir=tmp_path,
        run_id="../run id/" + "x" * 200,
        scene_id="../../scene id/" + "y" * 200,
        epoch=7,
    )

    _, run_component, file_component = pointer.split("/")
    assert len(run_component) <= 120
    assert len(file_component) <= 120
    assert re.fullmatch(r"[A-Za-z0-9._-]+", run_component)
    assert re.fullmatch(r"[A-Za-z0-9._-]+", file_component)
    assert (tmp_path / pointer).is_file()


def test_write_trial_sidecar_caps_complete_filename_with_huge_epoch(tmp_path: Path) -> None:
    """Catch an epoch suffix bypassing the complete path-component cap."""
    huge_epoch = 10**500
    pointer = write_trial_sidecar(
        [{"schema_version": 1}],
        log_dir=tmp_path,
        run_id="run",
        scene_id="scene",
        epoch=huge_epoch,
    )
    other_pointer = write_trial_sidecar(
        [{"schema_version": 1}],
        log_dir=tmp_path,
        run_id="run",
        scene_id="scene",
        epoch=huge_epoch + 1,
    )

    file_component = pointer.rsplit("/", 1)[1]
    other_component = other_pointer.rsplit("/", 1)[1]
    assert pointer != other_pointer
    assert len(file_component) <= 120
    assert len(other_component) <= 120
    assert re.fullmatch(r"scene-eh[0-9a-f]{16}\.jsonl", file_component)
    assert re.fullmatch(r"scene-eh[0-9a-f]{16}\.jsonl", other_component)
    assert re.fullmatch(r"[A-Za-z0-9._-]+", file_component)
    assert (tmp_path / pointer).is_file()
    assert (tmp_path / other_pointer).is_file()


def test_write_trial_sidecar_reserves_distinct_epoch_suffix_for_long_scene(
    tmp_path: Path,
) -> None:
    """Catch long scene truncation causing distinct trial epochs to overwrite one path."""
    scene_id = "scene-" + "x" * 300

    first = write_trial_sidecar(
        [{"epoch": 7}],
        log_dir=tmp_path,
        run_id="run",
        scene_id=scene_id,
        epoch=7,
    )
    second = write_trial_sidecar(
        [{"epoch": 8}],
        log_dir=tmp_path,
        run_id="run",
        scene_id=scene_id,
        epoch=8,
    )

    assert first != second
    first_name = first.rsplit("/", 1)[1]
    second_name = second.rsplit("/", 1)[1]
    assert first_name.endswith("-e7.jsonl")
    assert second_name.endswith("-e8.jsonl")
    assert len(first_name) <= 120
    assert len(second_name) <= 120
    assert (tmp_path / first).is_file()
    assert (tmp_path / second).is_file()
