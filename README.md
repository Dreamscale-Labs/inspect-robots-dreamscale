# inspect-robots-dreamscale

An [Inspect Robots](https://github.com/robocurve/inspect-robots) policy adapter for
Dreamscale-hosted DreamZero-YAM. Discovery and construction are offline; the first trial reset
opens one lazy Dreamscale connection, and later trials reuse that connection while starting fresh
logical episodes.

Licensed under Apache 2.0. It supports Python 3.11 through 3.14 and requires the immutable
`dreamscale[dreamzero]==0.1.0a26` SDK release.

Using it against a Dreamscale-hosted model needs an API key and an entitlement for that model;
the adapter itself is open.

Worked examples live in [`examples/`](examples/): a complete evaluation and a
skeleton embodiment showing the observation and action contract.

## Install and discover

```bash
uv add inspect-robots-dreamscale
uv run dreamscale login
```

`dreamscale login` opens a browser to approve this machine. On a headless controller, create a key
in the dashboard and run `uv run dreamscale login --api-key "<your key>"` instead.

Confirm the expected Dreamscale SDK is active before starting an evaluation:

```bash
uv run python -c 'import dreamscale; assert dreamscale.__version__ == "0.1.0a26"'
```

Verify that the entry point is available without opening a cloud session:

```bash
inspect-robots list policies
```

The output must contain `dreamscale`. Keep your existing registered task and embodiment; do not
replace or rename either. Change only the policy selection in your existing evaluation command:

```bash
--policy dreamscale -P model=dreamzero-yam
```

The default is YAM's qualified `async_latest` mode. Use `sampling=async_8` for the explicit
compatibility/rollback path and `sampling=upstream_eval` only for an open-loop dataset
evaluation. The server keeps inference single-flight and latest-only. The SDK preserves two
committed steps and applies its fixed absolute-target motion smoother only to the aligned
`async_latest` suffix; the adapter exposes no custom buffering, horizon, or smoothing knobs.

DreamZero-YAM is qualified at exactly 30 Hz. `-P control_hz=30` is accepted for explicitness;
every other value fails during policy construction, before a paid session is opened. The model's
30 Hz action/data timebase is distinct from inference frequency and the robot driver's internal
servo loop. Dynamic cadence must not be advertised until observation production, temporal
admission, inference, and action execution consume one resolved rate end to end.

`-P keep_warm_s=<seconds>` holds the session after close (0-3600, default 0) so the next run reclaims
it instead of starting cold -- 147s against 23s, measured back to back. **A hold is billed at the full
rate and `close()` no longer stops the meter**, because parking keeps the GPU reserved for you. Use it
while iterating; leave it at 0 for unattended runs.

**Nothing enforces this rate.** Inspect's rollout adds no wall-clock pacing, so the real rate is
however fast your embodiment's `step()` returns; `control_hz` is what the action scheduler plans
against. Pace the embodiment at 30 Hz. The adapter measures the gap between policy steps, records it
as `step_interval_ms` in the sidecar, and warns once if the measured rate diverges from 30 Hz by
more than 25%.

The first connection has a 1,800-second startup budget by default so DreamZero-YAM can finish
loading and warmup. Set `-P startup_timeout_s=<seconds>` to another finite positive value when a
target needs a different startup budget. This does not change `timeout_s`, the existing
per-step action deadline, which remains 60 seconds by default.

For a non-commanding integration preflight, call `prepare()` first. It opens or reuses the lazy
connection without starting an episode or inference, allowing a time-sensitive observation to be
captured only after a cold worker is ready. Then `predict_model_action(observation,
instruction=...)` blocks for one real model chunk without starting an execution episode. This
distinction matters in `async_latest`: the first `act()` may correctly be a hold while inference is
in flight, whereas the preflight method cannot pass until it has a model action. The caller must
validate and discard that action; a later `reset()` reuses the same Dreamscale connection for the live
Inspect episode.

## Observation and simulator contract

The existing task and embodiment must provide all of the following on every policy step:

- `top_cam`, `left_cam`, and `right_cam` uint8 images, each with its own real Unix-epoch capture
  time in seconds (not a process-monotonic clock and not one synthetic shared time);
- finite packed `joint_pos` state with shape `(14,)` in YAM left-arm, left-gripper, right-arm,
  right-gripper order; and
- Inspect's integer `extra["env_step"]`, starting at zero and advancing once per delivered action.

The adapter declares a 14-dimensional raw absolute-joint action at the commanded rate. It returns exactly one
action per Inspect `act()` call while Dreamscale owns DreamZero's managed action buffering. Simulator
compatibility means matching those camera, state, action, clock, and rate contracts; it does not by
itself establish physics parity, task success, or physical-robot safety.

## Artifact ownership and joining

Inspect remains canonical for the EvalLog, aggregate scores, post-approval commanded-action JSONL,
stored frames, Rerun recording, operator judgement, and trial termination/error state. The adapter
adds one atomic diagnostics sidecar and records its relative path at
`TrialRecord.metadata["dreamscale_telemetry"]`:

```text
dreamscale/<run_id>/<sanitized-scene-id>-e<epoch>.jsonl
```

Schema-v2 sidecar rows contain package versions, session and serving identity, timestamp source,
commanded cadence, model/hold action source, source control tick, source camera
capture-to-execution age, timing, accurate maximum overlapping-target revision, chunk/merge
disposition, and the same Inspect environment step. Join them to the EvalLog, action JSONL, or
Rerun timeline using `env_step`; use `join_key`
(`<cache_generation>:<logical_action_index>`) for Dreamscale chunk diagnostics. Sidecars do not
duplicate action vectors, images, credentials, authorization material, certificates, or endpoints.

## Deterministic cleanup

Evaluation owners must call `policy.close()` in `finally` after the run, even when Inspect reports an
error or cancellation. `close()` is synchronous and idempotent, and `policy.session_id` remains
readable after close so the caller can verify that the exact session is gone:

```bash
uv run dreamscale sessions list
```

Do not stop unrelated sessions. The adapter also registers a bounded process-exit fallback, but it
is not a substitute for explicit close.

## Physical YAM boundary

This integration does not authorize an unattended physical run. For any physical YAM test, you own
and must supply the embodiment package for your arm, validated limits, an attended operator gate, a
working e-stop, and a rehearsed termination procedure. This adapter calls neither the embodiment nor
hardware directly, and compatibility or serving evidence must not be reported as physical safety or
effectiveness evidence.
