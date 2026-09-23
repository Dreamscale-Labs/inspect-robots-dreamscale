# Examples

Reference code for running Dreamscale-hosted **DreamZero-YAM** on a physical
bimanual YAM through Inspect Robots, using this adapter.

See the [Inspect Robots guide](https://docs.dreamscalelabs.com/guides/inspect-robots)
for the full integration reference.

| directory | command rate | when to start here |
| --- | --- | --- |
| [`native-30hz/`](native-30hz/) | exactly 30 Hz | The only currently qualified DreamZero-YAM cadence |

The directory contains two files:

| file | what it is |
| --- | --- |
| `run_eval.py` | A complete evaluation: resolve the policy, run, close, then check where the actions came from |
| `yam_embodiment.py` | The observation and action contract, as a working skeleton |

## Cadence boundary

DreamZero-YAM's current action/data timebase is exactly 30 Hz. It is not the
model-inference call rate and it is not the I2RT driver's internal servo loop.
Earlier 5–30 Hz examples only changed the action scheduler's declared rate; they
did not prove a dynamic observation-to-action contract and are intentionally no
longer offered. A high hold fraction is a diagnostic to investigate, not a
reason to silently stretch the learned trajectory.

## Setup

```bash
uv add inspect-robots-dreamscale
uv run dreamscale login                 # or `dreamscale login --api-key "<your key>"` when headless
uv run dreamscale status --model dreamzero-yam
```

The SDK reads its credential from `~/.dreamscale/config.toml`, which `login`
writes.

Then edit `TASK_NAME` and `EMBODIMENT_NAME` in `run_eval.py` to your registered
names. Your task and embodiment do not change to use Dreamscale — only the policy
does.

> **These examples are deliberately incomplete, and none of it has run on a
> physical YAM.** The runs behind them used a stand-in arm: a pinned recording of
> a real YAM episode for the cameras, and a perfect-tracking executor. What is
> proven is the *serving* path — install, auth, session lifecycle, observation
> upload, action delivery, timing — which is independent of the robot.
>
> The hardware half is yours, and we cannot anticipate your setup: camera
> drivers and their real capture clocks, CAN bus and joint ordering, calibration,
> limits, and the gripper convention on your arm. `yam_embodiment.py` is a
> skeleton of the *contract*, not a driver. Expect to write real code where it
> says so, and expect the first hardware bring-up to surface things this file
> does not mention.

## The three things that actually bite

**1. A green run can contain no model actions.** Inspect reports success when a
trial completes without raising, and `episode_length` counts steps rather than
inference. An evaluation where the robot held position for every step looks
identical in the EvalLog to one the model drove. We shipped exactly that result
before checking. `run_eval.py` prints `action_source` after every run for this
reason; if you take one thing from these examples, take that check.

**2. Your control loop must pace itself, at the rate you declared.** Nothing
enforces the command rate: Inspect's rollout adds no wall-clock pacing, so your
embodiment is the clock. `control_hz` only tells the action scheduler what to
plan against. A loop that steps as fast as the code allows finishes the episode
before any chunk can arrive, and every action is a hold; a loop that steps at a
rate other than the one it declared makes every replan decision wrong. Physical
hardware paces itself for free; anything simulated or stubbed has to be told to.
The adapter measures the gap between steps and warns if it drifts more than 25%
from what you commanded.

**3. Keep `observe()` cheap.** Dreamscale samples observations on the *capture*
clock, not once per action, because the sampler always wants the freshest frame.
An `observe()` that takes anything like a frame period starves the loop that
receives actions. A 31 ms observation callback on a 33 ms budget was enough to
stall a run completely, with no error — it simply stopped delivering.

## Reading the numbers

The adapter writes a sidecar per trial at
`logs/dreamscale/<run_id>/<scene>-e<epoch>.jsonl`, and records its path at
`TrialRecord.metadata["dreamscale_telemetry"]`. Server-stamped fields arrive with
the action, so they are measurements rather than client guesses:

- `server_inference_ms` — GPU time
- `obs_to_action_ms` — the full round trip
- `action_source` — `model` or `hold`, per step
- `step_interval_ms` — the gap between policy steps, i.e. the rate you *actually* ran at

`obs_to_action_ms − server_inference_ms` is network, encode, decode, and
scheduling combined. It is a residual, not a measurement, and it is dominated by
distance to `us-west-2`. From Sydney it was ~270–310 ms p50; co-locating the
client near the region moves it far more than any client-side tuning.

Sidecars carry no action vectors, images, credentials, or endpoints, so they are
safe to attach to a bug report.

## Safety

These examples exercise the software path. They do not authorize an unattended
physical run. Validated joint limits, an attended operator gate, a working
e-stop, and a rehearsed termination procedure remain yours. Serving evidence is
not safety evidence.
