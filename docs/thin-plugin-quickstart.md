# DreamZero-YAM from a blank rig computer: Inspect Robots + Dreamscale (preview)

> **This is a preview, not the supported path.** It describes the `thin-plugin` branch of
> `inspect-robots-dreamscale` (version `0.2.0.dev0`). That version is not released and not on PyPI,
> and it has not been qualified on real hardware yet.
>
> The supported setup is still the rig composition,
> [`inspect-robots-dreamscale-yam`](https://github.com/Dreamscale-Labs/inspect-robots-dreamscale-yam)
> `stable` (v0.1.21). Keep this preview in its own folder and virtual environment. Never run both
> setups on the same rig at the same time: they use different locks and can't see each other.

## What this setup is

You use Robocurve's [Inspect Robots](https://github.com/robocurve/inspect-robots) and its
[YAM package](https://github.com/robocurve/inspect-robots-yam) exactly as they ship. Dreamscale adds
one policy, `dreamscale`, which runs DreamZero-YAM on Dreamscale's cloud GPUs.

These all come from Inspect Robots, unchanged:

- hardware setup
- safety guardrails
- runs, epochs and eval sets
- operator grading
- logs and the viewer

Dreamscale is responsible for three things:

1. the GPU session: cold start, reuse across rollouts, and a warm hold between processes;
2. sign-in and model access;
3. per-step timing telemetry written into your log folder.

## 0. Before you start

- **Computer:** Ubuntu or Debian Linux, connected to both I2RT YAM arms by USB-CAN and to three
  cameras: a RealSense D435 on top and a D405 on each wrist.
- **Account:** a Dreamscale account with DreamZero-YAM access. Check it at
  [app.dreamscalelabs.com](https://app.dreamscalelabs.com).
- **Safety:** an e-stop within reach for every step that moves the arms.

## 1. System packages (one sudo step)

```bash
sudo apt-get update && sudo apt-get install -y build-essential can-utils cmake curl git libgl1 libglib2.0-0 libusb-1.0-0-dev ninja-build pkg-config python3-dev v4l-utils
```

## 2. Install uv, Inspect Robots + YAM, the arm driver and Dreamscale

Skip the first command if `uv --version` already works.

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh && source ~/.local/bin/env
```

```bash
mkdir -p ~/dreamscale-rig && cd ~/dreamscale-rig && uv venv -p 3.12 && source .venv/bin/activate
```

Install Inspect Robots and the YAM package. The `depth` extra lets the D405 wrist cameras be read
through librealsense. This brings in Inspect Robots 0.59.

```bash
uv pip install "inspect-robots-yam[depth]==0.36.0" "inspect-robots[rerun]"
```

Install the I2RT arm driver. It is only on GitHub; the build constraint works around a build issue in one of its
dependencies, and matches upstream YAM's instructions:

```bash
echo 'scikit-build-core<0.10' > build-constraints.txt && uv pip install --build-constraints build-constraints.txt "i2rt @ git+https://github.com/i2rt-robotics/i2rt@ac096928d6899ddf852a71c5e8fbaa6055cd9745"
```

Install the Dreamscale policy from the preview branch. You need `--prerelease allow` because both
the preview and the Dreamscale SDK are pre-release versions.

```bash
uv pip install --prerelease allow "inspect-robots-dreamscale @ git+https://github.com/Dreamscale-Labs/inspect-robots-dreamscale@thin-plugin"
```

## 3. Sign in to Dreamscale

```bash
dreamscale login
```

It prints a link and a code. Open the link on any device, sign in and approve. On a computer with
no browser, you can instead create a key in the dashboard and run
`dreamscale login --api-key "<key>"`.

Confirm that your account can use the model:

```bash
dreamscale status --model dreamzero-yam
```

## 4. Hardware setup with Inspect Robots' wizard

Check that both CAN interfaces are UP:

```bash
ip -brief link show type can
```

If one shows `DOWN`, bring it up. Replace `can0` with its name:

```bash
sudo ip link set can0 up type can bitrate 1000000
```

Run the wizard:

```bash
inspect-robots setup
```

- **CAN and cameras:** it asks which camera is top, left and right, and which CAN channel drives
  each arm. It can identify each one when you unplug it.
- **udev rules:** it prints rules that pin each CAN adapter's name by serial number. Install them
  unless your rig already has its own CAN naming rules. Without pinned names, `can0` and `can1` can
  swap after a reboot.
- **Options:** answer **no** to the collision guardrail and to `auto_start` for now.
- **Output:** it writes `~/.config/inspect-robots/config.ini`.

## 5. Point the config at Dreamscale

Edit `~/.config/inspect-robots/config.ini`. Add or change the lines below, and keep the camera and
CAN lines the wizard wrote.

```ini
[defaults]
policy = dreamscale
embodiment = yam_arms
max_steps = 3600          ; 120 s at 30 Hz (the built-in default of 300 is only 10 s at 30 Hz)
scorer = operator         ; score the y/n verdict you type (success_at_end would score 0 on a real rig)
store_frames = true

[policy.args]
model = dreamzero-yam

[embodiment.args]
control_hz = 30
cam_width = 640
cam_height = 360
auto_start = false        ; ask before homing, instead of moving as soon as the run starts
```

DreamZero-YAM is qualified only at **30 Hz with 640×360 frames**. Stock YAM defaults to 10 Hz and
224×224, and without these three lines the run is refused before anything moves.

**Optional: read the wrist cameras through librealsense** (Robocurve's usual layout).

1. In `[embodiment.args]`, delete the `left_cam_device` and `right_cam_device` lines.
2. Add these, with the serial numbers **in quotes**:

   ```ini
   left_depth_serial = "YOUR-LEFT-D405-SERIAL"
   right_depth_serial = "YOUR-RIGHT-D405-SERIAL"
   ```

Find the serials with `rs-enumerate-devices`. A camera slot uses either `*_cam_device` or
`*_depth_serial`, never both.

## 6. Checks before anything moves

Confirm that the policy and the robot agree on actions, cameras and state. Nothing moves:

```bash
inspect-robots-yam-preflight --dry-run
```

Check sign-in, model access, GPU availability and the network:

```bash
dreamscale doctor
```

Aim the cameras. This opens a live page at `http://<rig-ip>:8807/` and never touches the motors;
press Ctrl-C when you are done:

```bash
inspect-robots-yam-health --watch
```

## 7. Rollouts

Hold the e-stop, and keep hands clear of the grippers.

Add `--max-action-delta 0.2` to every run command below, as the one-rollout example does. It limits each joint to
0.2 rad of change per 30 Hz step, the same cap the qualified v0.1.21 rig uses. Without it, stock
Inspect Robots allows about 5% of each joint's range per step, which is up to about 0.3 rad on
some YAM joints. It also doesn't limit the first action of a rollout. `max_action_delta` isn't a
`config.ini` setting, so pass it on the command line, or put it in the `./run` wrapper below.

### One rollout

```bash
inspect-robots "Pack container" --max-action-delta 0.2
```

1. The first run cold-starts the GPU, which takes a few minutes. While it starts, a spinner shows
   the elapsed seconds and the current startup stage. When output is not a terminal, it prints
   plain lines and a "still starting" line every 30 s. A final line says the session is ready.
2. YAM asks you to stand clear for homing, then to set up the scene. Press Enter each time.
3. The rollout runs for up to 120 s. Press **Esc** to end it early.
4. The arms park, and you answer `did the robot succeed? [y/n/partial/skip]`, plus an optional note.

### The same task several times in one process

```bash
inspect-robots "Pack container" --epochs 5
```

One GPU session serves all five rollouts. Between rollouts the arms stay powered at home while you
reset the scene.

### Robocurve's batch style: one process per rollout

In this style the arms are unpowered between rollouts. Set it up once:

```bash
git clone --depth 1 -b v0.36.0 https://github.com/robocurve/inspect-robots-yam ~/inspect-robots-yam
```

```bash
cp ~/.config/inspect-robots/config.ini ~/dreamscale-rig/config.ini && printf '#!/usr/bin/env bash\nexec ~/dreamscale-rig/.venv/bin/inspect-robots run --config "$(dirname "$0")/config.ini" --max-action-delta 0.2 "$@"\n' > ~/dreamscale-rig/run && chmod +x ~/dreamscale-rig/run
```

Then run a batch:

```bash
cd ~/dreamscale-rig && ~/inspect-robots-yam/scripts/run_batch.sh -n 10 --instruction "Pack container"
```

- Each rollout is a new process.
- The GPU session stays warm for 5 minutes after each process exits, so the next rollout picks it up
  in seconds instead of starting cold.
- The script asks you to reset the scene between rollouts, and saves verdicts to `logs/batches/`.

### Different tasks

- Change the instruction on your next command.
- Or run several registered tasks in one go: `inspect-robots eval-set 'my-bench/*'`.
- The GPU session carries over in both cases.

### Browse the results

```bash
inspect-robots view logs/ --serve --host 0.0.0.0 --open
```

This shows every rollout, verdict, note and camera frame. Dreamscale's per-step timing for each
trial is in `logs/dreamscale/`, linked from that trial's metadata.

## 8. When you're done

The GPU session stays warm, and **billed**, for 5 minutes after each process exits. On your last
run, add `-P keep_warm_s=0` to stop the session as soon as the process exits. Or stop it afterwards:

```bash
dreamscale sessions list
```

```bash
dreamscale sessions stop <session-id>
```

## Good to know

- **If a motor stops responding in the middle of a run:** you'll see `fail to communicate with the
  motor N`. Stock YAM keeps the episode running with that arm dead, so press **Esc** at once. Then
  check that arm's CAN cable and power, and run `ip -details -statistics link show <channel>`. A
  healthy bus shows `ERROR-ACTIVE` with zero `error-warn` and `error-pass`.
- **Updating the preview:** the branch changes. To pick up the latest commit, run:
  `uv pip install --prerelease allow --reinstall-package inspect-robots-dreamscale "inspect-robots-dreamscale @ git+https://github.com/Dreamscale-Labs/inspect-robots-dreamscale@thin-plugin"`.

- **Warm hold:** change it with `-P keep_warm_s=<0..3600>`, or set it in `[policy.args]`. The
  default is 300 s.
- **Ctrl-C and SIGTERM:** while a GPU session is open, both cancel the run cleanly. Inspect Robots
  then parks the arms, and the adapter parks or stops exactly the session this process opened.
- **Camera timing:** stock YAM doesn't record when each camera captured its frame, so the adapter
  timestamps each frame when it arrives. Each telemetry row records this as
  `capture_timing: observation_fallback`. A camera that freezes without reporting an error is not
  detected in this mode.
- **Logs:** Inspect Robots 0.53 can't read logs written by 0.59. Keep this setup's `logs/` separate
  from the v0.1.21 rig's logs.

## Appendix: the a7 test rig (Dreamscale internal)

Checked on 2026-09-23.

- **Already done:** all of step 1, and uv is installed (0.12.18). Start at step 2 with the
  `mkdir -p ~/dreamscale-rig` line.
- **Sign-in:** a7 is already signed in as `chrisyooak@gmail.com`, from the v0.1.21 rehearsal. Run
  `dreamscale status --model dreamzero-yam` to confirm. If you run `~/rehearsal-stash-*/restore.sh`
  later, it moves that sign-in aside, and the SDK falls back to the old `~/.dropbear` test-account
  key. Run `dreamscale login` again after restoring.
- **CAN:** already UP and named `can_follower_l` / `can_follower_r` by `/etc/udev/rules.d/90-can.rules`.
  Pick those names in the wizard, and **do not** install the udev rules it prints.
- **Cameras:**

  | Role | Camera | Serial |
  | --- | --- | --- |
  | top | D435 | `146322072458` |
  | left wrist | D405 | `261022277065` |
  | right wrist | D405 | `261022277669` |

  The D405 with serial `261922270754` belongs to the `yam-rollout-recorder` service. Never assign
  it.
- **Running both setups:** don't run the v0.1.21 rig (`./dreamscale-yam run`) and this setup at the
  same time.
