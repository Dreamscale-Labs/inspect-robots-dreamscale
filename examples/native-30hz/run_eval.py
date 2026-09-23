"""Run a Dreamscale-hosted DreamZero-YAM policy through Inspect Robots.

Adapted from a run that actually completed against production:
120 steps, 51 model-sourced actions, GPU inference 324.9 ms p50, over a
`wss_tunnel` data plane to `us-west-2`.

Replace `TASK_NAME` and `EMBODIMENT_NAME` with your own registered names. Your
task and embodiment do not change to use Dreamscale -- only the policy does.

    python run_eval.py

Requires `dreamscale login` (or `dreamscale login --api-key "<key>"` on a headless
controller) and an entitlement for the model.
"""

from __future__ import annotations

import collections
import json
import pathlib
import sys

from inspect_robots import eval as run_evaluation
from inspect_robots.registry import resolve

MODEL = "dreamzero-yam"
TASK_NAME = "your-task"
EMBODIMENT_NAME = "your-embodiment"
LOG_DIR = "logs"

# DreamZero-YAM is currently qualified at exactly 30 Hz. Nothing enforces the
# actual loop cadence: Inspect adds no pacing, so the embodiment must deliver
# 30 Hz. The adapter records measured step intervals and warns on divergence.
CONTROL_HZ = 30


def summarise_sidecar(log_dir: str) -> None:
    """Report where the actions actually came from.

    This is the check worth running on every evaluation. Inspect reports a trial
    green when it completes without raising, and `episode_length` counts steps
    rather than inference, so a run in which the robot held position for every
    step is indistinguishable in the EvalLog from one the model drove. The
    difference only appears here.
    """

    sidecars = sorted(pathlib.Path(log_dir).glob("dreamscale/*/*.jsonl"))
    if not sidecars:
        print("no Dreamscale sidecar found; did the policy connect?", file=sys.stderr)
        return

    rows = [json.loads(line) for line in sidecars[-1].read_text().splitlines() if line.strip()]
    sources = collections.Counter(row.get("action_source") for row in rows)
    print(f"\naction sources: {dict(sources)}")
    if not sources.get("model"):
        print(
            "WARNING: no model-sourced actions. The robot held position for the whole\n"
            "episode. The usual cause is a control loop faster than one round trip, so\n"
            "no chunk arrives before the episode ends.",
            file=sys.stderr,
        )

    # The rate the loop actually ran at, which nothing enforces. If this is far
    # from CONTROL_HZ, the action scheduler has been planning against a rate the
    # robot was not running at.
    intervals = sorted(
        row["step_interval_ms"] for row in rows if isinstance(row.get("step_interval_ms"), float)
    )
    if intervals:
        median_ms = intervals[len(intervals) // 2]
        print(f"measured rate: {1000.0 / median_ms:.1f} Hz (commanded {CONTROL_HZ})")

    timings = [entry for row in rows if row.get("timing") for entry in row["timing"]]
    if not timings:
        return

    def p50(key: str) -> float | None:
        values = sorted(e[key] for e in timings if isinstance(e.get(key), (int, float)))
        return values[len(values) // 2] if values else None

    total, gpu = p50("obs_to_action_ms"), p50("server_inference_ms")
    print(f"replans: {len(timings)}")
    print(f"  observation to action p50: {total:.0f} ms")
    print(f"  GPU inference       p50: {gpu:.0f} ms")
    if total is not None and gpu is not None:
        # A residual: network, encode, decode, and scheduling together.
        print(f"  client + network    p50: {total - gpu:.0f} ms (residual)")


def main() -> int:
    # No credential preflight here on purpose. The SDK reads its key from
    # ~/.dreamscale/config.toml, not from the environment, so an env-var check
    # would pass for someone who has not run `dreamscale login` and fail for
    # someone who has. `connect()` already reports a missing credential
    # precisely, and names the command that fixes it.

    # `resolve` returns a constructed object, not a factory. Policy parameters
    # are the same ones you would pass with `-P`, given as keyword arguments.
    policy = resolve("policy", "dreamscale", model=MODEL, control_hz=CONTROL_HZ)
    task = resolve("task", TASK_NAME)
    embodiment = resolve("embodiment", EMBODIMENT_NAME)

    try:
        logs = run_evaluation(
            task=task,
            policy=policy,
            embodiment=embodiment,
            log_dir=LOG_DIR,
        )
    finally:
        # Sessions bill while open and hold an exclusive lease. Closing is not
        # optional, including when the run raised.
        policy.close()

    log = logs[0] if isinstance(logs, (list, tuple)) else logs
    print(f"status: {log.status}")
    for sample in getattr(log, "samples", None) or []:
        print(f"  {sample.scene_id}: {sample.status}  error={sample.error}")

    summarise_sidecar(LOG_DIR)
    return 0 if getattr(log, "status", None) == "success" else 1


if __name__ == "__main__":
    sys.exit(main())
