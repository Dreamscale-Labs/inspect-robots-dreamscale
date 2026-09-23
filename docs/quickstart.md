# Quickstart: DreamZero-YAM through Inspect Robots

Go from a Linux rig computer with nothing installed to DreamZero-YAM running a rollout on a bimanual
I2RT YAM, using Robocurve's [Inspect Robots](https://github.com/robocurve/inspect-robots) and its
stock [YAM package](https://github.com/robocurve/inspect-robots-yam), with Dreamscale as the policy.

Stack:

| Package | Version |
| --- | --- |
| `inspect-robots` | 0.59.0 |
| `inspect-robots-yam` | 0.36.0 |
| `i2rt` | `ac09692` |
| `inspect-robots-dreamscale` | v0.2.0 |
| `dreamscale` SDK | 0.1.0a26 |

You need a Dreamscale account with DreamZero-YAM access, and enough credit.
DreamZero-YAM bills by session time, which includes cold starts and the 5-minute warm hold after
each run.

Each step says **where** to run it:

- **[rig]**: a terminal on the rig computer.
- **[phone/laptop]**: any device with a browser.

If you open a new terminal partway through, run this first:

```bash
cd ~/dreamscale-rig && source .venv/bin/activate
```

The example task is "Pick up the yellow block and place it into the blue bowl". Replace it with
your own instruction.

## Part B. Install and set up (on your rig)

### B0. Physical checklist

Before running anything:

- Both YAM arms are powered.
- Both USB-CAN adapters are plugged into the rig computer.
- All three cameras are plugged in: the D435 on top, and a D405 on each wrist.
- The e-stop is in reach.
- No other program is using the arms or cameras. Stop any Inspect Robots run you have open.

### B1. [rig] System packages

This needs your sudo password.

```bash
sudo apt-get update && sudo apt-get install -y build-essential can-utils cmake curl git libgl1 libglib2.0-0 libusb-1.0-0-dev ninja-build pkg-config python3-dev v4l-utils
```

### B2. [rig] Install uv

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

```bash
source $HOME/.local/bin/env
```

Check it:

```bash
uv --version
```

This prints a version, for example `uv 0.12.x`.

### B3. [rig] Create the Dreamscale rig folder and its environment

Everything goes in `~/dreamscale-rig`. That includes this environment's own Inspect Robots config
file, so any Inspect Robots setup you already have (`~/.config/inspect-robots/config.ini`) is not
touched.

```bash
mkdir -p ~/dreamscale-rig && cd ~/dreamscale-rig
```

```bash
uv venv -p 3.12
```

```bash
echo 'export INSPECT_ROBOTS_CONFIG="$HOME/dreamscale-rig/config.ini"' >> .venv/bin/activate
```

```bash
source .venv/bin/activate
```

Check that the prompt now starts with `(dreamscale-rig)`, and that this prints
`/home/<user>/dreamscale-rig/config.ini`:

```bash
echo $INSPECT_ROBOTS_CONFIG
```

### B4. [rig] Install Inspect Robots, YAM, the arm driver and Dreamscale

Run the four commands in order. Each takes a minute or two.

```bash
uv pip install "inspect-robots-yam[depth]==0.36.0" "inspect-robots[rerun]==0.59.0"
```

```bash
echo 'scikit-build-core<0.10' > build-constraints.txt
```

```bash
uv pip install --build-constraints build-constraints.txt "i2rt @ git+https://github.com/i2rt-robotics/i2rt@ac096928d6899ddf852a71c5e8fbaa6055cd9745"
```

```bash
uv pip install "inspect-robots-dreamscale @ git+https://github.com/Dreamscale-Labs/inspect-robots-dreamscale@v0.2.0" "dreamscale[dreamzero]==0.1.0a26"
```

Check that Inspect Robots can see both pieces:

```bash
inspect-robots list
```

The output must list `yam_arms` under `embodiments:` and `dreamscale` under `policies:`.

### B5. [rig + phone/laptop] Sign in

[rig]:

```bash
dreamscale login
```

It prints `Sign in to Dreamscale:`, a URL and a `Code:`, then `Waiting for approval...`.

[phone/laptop]:

1. Open the URL.
2. Check that the code shown matches the one in the terminal.
3. Sign in with your Dreamscale account.
4. Click **Approve Dreamscale CLI**.

[rig]: the terminal confirms the sign-in and returns to the prompt.

### B6. [rig] Check the account and the service

```bash
dreamscale status --model dreamzero-yam
```

This shows DreamZero-YAM and its region capacity.

```bash
dreamscale doctor
```

Every check passes.

### B7. [rig] Check both CAN interfaces are up

```bash
ip -brief link show type can
```

This prints two lines, one per USB-CAN adapter, for example `can0` and `can1`, or custom names. Both
must say `UP`. If one says `DOWN`, run this with that interface's name in place of `can0`, then
run the check again:

```bash
sudo ip link set can0 up type can bitrate 1000000
```

### B8. [rig] Run the Inspect Robots setup wizard

```bash
inspect-robots setup
```

The first line must say it writes `/home/<user>/dreamscale-rig/config.ini`. Each prompt shows a
suggested value in `[brackets]`; pressing Enter accepts it. Answer as follows.

**Defaults.** Type these exactly:

| Prompt | Type |
| --- | --- |
| `policy` | `dreamscale` |
| `embodiment` | `yam_arms` |
| `scorer` | `operator` |
| `max steps` | `3600` |
| `live rerun viewer` | `false` |
| `store camera frames` | `true` |

**Devices.** If it asks `Configure devices?`, answer `y`. It then asks for five devices, one at a
time, each with a numbered list:

1. `left arm CAN channel`
2. `right arm CAN channel`
3. `top camera`
4. `left camera`
5. `right camera`

For each one, identify the device by unplugging it:

1. Type `u` and press Enter.
2. When it says to, unplug that device's USB cable. For a CAN channel, that's the arm's USB-CAN
   adapter; for a camera, that camera's USB cable.
3. Plug it back in when it says to.
4. It prints `That was: <name>` and moves on.

"Left" and "right" are from the robot's point of view, matching your existing rig setup.

**CAN name rules.** After the devices, it may print a block of udev rules to paste into
`/etc/udev/rules.d/70-can-names.rules`. **Don't install them today.** Instead, from the end of
the wizard until the onboarding is over, don't unplug the USB-CAN adapters and don't reboot the rig. Pinning the CAN names is a
follow-up.

**Options.** Answer each yes/no question as shown:

| Prompt starts with | Answer | Why |
| --- | --- | --- |
| `Skip the operator start prompts (auto_start)` | `n` | The rig waits for Enter before homing and before each rollout. |
| `Block predicted arm collisions before they happen` | `n` | This needs measured arm-base positions, which we don't have yet. |
| `Report estimated joint effort in observations` | `n` | |
| `Open EEF pitch/roll tilt axes` | `n` | |
| `Motor temperature soft limit in degrees C` | press Enter (accepts `70`) | |

It finishes with `Wrote /home/<user>/dreamscale-rig/config.ini`.

### B8b. [rig] Re-check CAN after the unplugging

Unplugging a USB-CAN adapter can bring it back `DOWN`. Run the B7 check again:

```bash
ip -brief link show type can
```

Both lines must say `UP`. For any line that says `DOWN`, run this with its name in place of
`can0`:

```bash
sudo ip link set can0 up type can bitrate 1000000
```

### B9. [rig] Add DreamZero-YAM's rate and camera size to the config

DreamZero-YAM is qualified only at 30 Hz with 640×360 camera frames. Stock YAM defaults to 10 Hz
and 224×224. **Nothing refuses a run without these lines:** preflight still passes, and the robot
would run the model at the wrong rate and image size. The `grep` check below is the safeguard.
Run this command **once**:

```bash
sed -i '/^\[embodiment.args\]/a control_hz = 30\ncam_width = 640\ncam_height = 360' ~/dreamscale-rig/config.ini
```

Check it. This must print exactly three lines: `control_hz = 30`, `cam_width = 640` and
`cam_height = 360`.

```bash
grep -E "^(control_hz|cam_width|cam_height)" ~/dreamscale-rig/config.ini
```

### B10. [rig] Preflight: nothing moves

```bash
inspect-robots-yam-preflight --dry-run
```

This must print both of these lines:

```
OK: policy and embodiment are compatible.
(dry-run) No motion will be commanded.
```

### B11. [rig + phone/laptop] Check the camera roles and aim

Get the rig's IP address:

```bash
hostname -I
```

Start the camera viewer. It never touches the motors.

```bash
inspect-robots-yam-health --watch
```

[phone/laptop]: open `http://<first IP from hostname -I>:8807/`. Check these three things, and
adjust the cameras until the scene is framed:

- The **top** stream is the D435 overhead view.
- The **left** stream is the left wrist.
- The **right** stream is the right wrist.

[rig]: press **Ctrl-C** to stop the viewer.

If a role is wrong, run `inspect-robots setup` again. Press Enter to keep every answer except the
wrong camera, then run this step again. You don't need to redo B9: the wizard keeps
`control_hz`, `cam_width` and `cam_height`.

---

## Part C. The first rollout

### C1. Set the scene

- Put the **yellow block** and the **blue bowl** on the table, inside the top camera's view.
- The operator holds the **e-stop**.
- Everyone keeps their hands clear of the grippers and arms.

### C2. [rig] Run the rollout

Run it from `~/dreamscale-rig`, so logs go to `~/dreamscale-rig/logs/`:

```bash
cd ~/dreamscale-rig && inspect-robots "Pick up the yellow block and place it into the blue bowl" --max-action-delta 0.2
```

`--max-action-delta 0.2` limits each joint to 0.2 rad of change per 30 Hz step. That is the same
cap the qualified Dreamscale rig uses. Use it on every run.

What you will see, in order, and what to do:

1. Header lines: `policy: dreamscale (from …/dreamscale-rig/config.ini)`,
   `embodiment: yam_arms (from …)`, a collision-guardrail warning (expected),
   `guardrails: clamp + delta-limit`, and the operator console help line.
2. `dreamscale: keep_warm_s=300: …`: the GPU stays warm, and billed, for 5 minutes after the run,
   so the next run starts in seconds.
3. `⠋ 1m12s dreamscale: starting compute (a few mins)`: the first cold start. It usually takes 2–8
   minutes. **Wait.**
4. `dreamscale: session dreamscale_sess_… ready in N s`.
5. `initializing motorchain robot …`, printed twice (once per arm).
6. `Arms will move to the home pose - stand clear, then press Enter...`: stand clear, then press
   **Enter**. The arms ramp to the start pose.
7. `Position the scene, then press Enter to start...`: put the block and bowl in their start
   positions, step back, then press **Enter**.
8. `Running. Max ~120s.`: the policy is driving. A line saying the embodiment supplies no
   per-camera `image_times` is expected with stock YAM.
   - Press **Esc** to end the rollout early, for example once the block is in the bowl, or if
     anything looks wrong.
   - Use the **e-stop** for anything unsafe.
9. `parking for grading: ramping arms clear`.
10. `did the robot succeed? [y/n/partial/skip]`: type `y` or `n` and press Enter. `partial` counts
    as a failure.
11. `grader notes (Enter for none):`: type a short note, or press Enter.
12. `parking: ramping arms back before torque-off`, then `Robot closed with all torques set to zero.`
    printed twice.
13. A summary ending `operator: 1` (success) or `operator: 0`, then `log: logs/adhoc_….json`.
14. `dreamscale: Run summary: …`, then `dreamscale: released session … to a warm hold of up to 300 s …`.

### C3. [rig] More rollouts

To run it again, run the same command within 5 minutes. The warm session is picked up in about
3 seconds instead of a cold start:

```bash
inspect-robots "Pick up the yellow block and place it into the blue bowl" --max-action-delta 0.2
```

To run five in a row in one process, with the arms staying powered at home while you reset the
scene between rollouts:

```bash
inspect-robots "Pick up the yellow block and place it into the blue bowl" --max-action-delta 0.2 --epochs 5
```

### C4. [rig] Look at the results

```bash
inspect-robots view logs/ --open
```

This opens an index of every run, with verdicts, notes and camera frames. Dreamscale's per-step
timing for each rollout is in `logs/dreamscale/`.

---

## Part D. Finish

### D1. [rig] Stop the GPU session

Run this when no more rollouts are planned:

```bash
dreamscale sessions list
```

If a session is listed, stop it, using the ID from the list:

```bash
dreamscale sessions stop <session-id>
```

Check that it's gone:

```bash
dreamscale sessions list
```

This prints `No open sessions.`.

## If something goes wrong

- **`fail to communicate with the motor N …`** during a rollout: press **Esc** at once. Stock YAM
  keeps the rollout running with that arm dead. Then check that arm's CAN cable and power, and
  check its bus with `ip -details -statistics link show <can-name>`. It should say
  `ERROR-ACTIVE`, with `error-warn` and `error-pass` at 0.
- **Motion looks sluggish, or the terminal warns that the measured step rate differs from 30 Hz:**
  check B9 with its `grep`. All three lines must be present.
- **The spinner runs for more than 15 minutes:** press **Ctrl-C**, then run
  `dreamscale status --model dreamzero-yam` and `dreamscale sessions list`.
- **A new terminal can't find `inspect-robots` or `dreamscale`:**
  `cd ~/dreamscale-rig && source .venv/bin/activate`.
