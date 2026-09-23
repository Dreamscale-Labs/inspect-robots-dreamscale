"""Warm-by-default, reclaim across processes, and bounded release at exit."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import textwrap
import threading
import time
from pathlib import Path
from typing import Any

import pytest
from inspect_robots.scene import Scene

import inspect_robots_dreamscale.policy as adapter_policy
from inspect_robots_dreamscale import dreamscale_policy

FAKE_CONTROL_PLANE = Path(__file__).with_name("fake_control_plane.py")
posix_only = pytest.mark.skipif(sys.platform == "win32", reason="POSIX signals and flock")


class ReleaseRecordingRemote:
    """Minimal remote whose close can be made slow to exercise the exit bound."""

    session_id = "sess-owned"

    def __init__(self, *, close_delay_s: float = 0.0, close_error: Exception | None = None):
        self.close_delay_s = close_delay_s
        self.close_error = close_error
        self.close_calls = 0
        self.closed = threading.Event()

    def begin_episode(self, **_kwargs: Any) -> None:
        return None

    def end_episode(self) -> None:
        return None

    def close(self) -> None:
        self.close_calls += 1
        time.sleep(self.close_delay_s)
        if self.close_error is not None:
            raise self.close_error
        self.closed.set()


def send_sigint() -> None:
    """A real SIGINT: unlike _thread.interrupt_main it wakes a blocked wait."""
    os.kill(os.getpid(), signal.SIGINT)


def connected_policy(monkeypatch, remote: ReleaseRecordingRemote, **kwargs: Any):
    monkeypatch.setattr(
        "inspect_robots_dreamscale.policy.dreamscale.connect",
        lambda *_args, **_kwargs: remote,
    )
    policy = dreamscale_policy(model="dreamzero-yam", **kwargs)
    policy.prepare()
    return policy


def test_keep_warm_defaults_to_300_and_says_it_is_billed(monkeypatch, capsys) -> None:
    """Catch a warm default that is either off or silent about its cost."""
    received: list[int] = []

    def connect(*_args: Any, keep_warm: int, **_kwargs: Any) -> ReleaseRecordingRemote:
        received.append(keep_warm)
        return ReleaseRecordingRemote()

    monkeypatch.setattr("inspect_robots_dreamscale.policy.dreamscale.connect", connect)
    policy = dreamscale_policy(model="dreamzero-yam")
    policy.prepare()
    policy.prepare()

    assert policy.keep_warm_s == 300
    assert received == [300]
    notices = [line for line in capsys.readouterr().err.splitlines() if "billed" in line]
    assert len(notices) == 1
    assert "-P keep_warm_s=0" in notices[0]


def test_keep_warm_zero_stops_and_prints_no_billing_notice(monkeypatch, capsys) -> None:
    """Catch `-P keep_warm_s=0` still parking, or still warning about a hold."""
    received: list[int] = []

    def connect(*_args: Any, keep_warm: int, **_kwargs: Any) -> ReleaseRecordingRemote:
        received.append(keep_warm)
        return ReleaseRecordingRemote()

    monkeypatch.setattr("inspect_robots_dreamscale.policy.dreamscale.connect", connect)
    policy = dreamscale_policy(model="dreamzero-yam", keep_warm_s=0)
    policy.prepare()
    policy.close()

    assert received == [0]
    err = capsys.readouterr().err
    assert "billed" not in err
    assert "stopped session sess-owned" in err


@pytest.mark.parametrize(("keep_warm_s", "said"), [(300, "warm hold"), (0, "stopped session")])
def test_exit_release_closes_exactly_the_owned_session_once(
    monkeypatch, capsys, keep_warm_s: int, said: str
) -> None:
    """Catch the exit path skipping release, releasing twice, or misreporting it."""
    remote = ReleaseRecordingRemote()
    policy = connected_policy(monkeypatch, remote, keep_warm_s=keep_warm_s)

    policy._atexit_close()
    policy._atexit_close()
    policy.close()

    assert remote.close_calls == 1
    assert policy.session_id == "sess-owned"
    err = capsys.readouterr().err
    assert said in err and "sess-owned" in err
    assert "could not confirm" not in err


def test_exit_release_is_bounded_when_close_hangs(monkeypatch, capsys) -> None:
    """Catch a wedged network holding interpreter exit open indefinitely."""
    remote = ReleaseRecordingRemote(close_delay_s=5.0)
    policy = connected_policy(monkeypatch, remote)
    policy.release_timeout_s = 0.3

    started = time.monotonic()
    policy._atexit_close()
    elapsed = time.monotonic() - started

    assert 0.25 <= elapsed < 1.5
    err = capsys.readouterr().err
    assert "could not confirm parking session sess-owned (timed out after 0.3 s)" in err
    assert "dreamscale sessions stop sess-owned" in err


def test_exit_release_reports_a_failed_close(monkeypatch, capsys) -> None:
    """Catch a failed park/stop being swallowed without naming the session."""
    remote = ReleaseRecordingRemote(close_error=RuntimeError("control plane 503"))
    policy = connected_policy(monkeypatch, remote, keep_warm_s=0)

    policy._atexit_close()

    err = capsys.readouterr().err
    assert "could not confirm stopping session sess-owned" in err
    assert "RuntimeError: control plane 503" in err
    assert "stops it when its 60 s lease lapses" in err


@posix_only
def test_first_interrupt_during_exit_release_is_absorbed(monkeypatch, capsys) -> None:
    """Catch a reflexive second Ctrl-C at exit abandoning an almost-finished release."""
    remote = ReleaseRecordingRemote(close_delay_s=0.6)
    policy = connected_policy(monkeypatch, remote)
    threading.Timer(0.1, send_sigint).start()

    policy._atexit_close()

    assert remote.closed.is_set()
    err = capsys.readouterr().err
    assert "interrupt again to abandon" in err
    assert "could not confirm" not in err


@posix_only
def test_second_interrupt_during_exit_release_abandons_the_wait(monkeypatch, capsys) -> None:
    """Catch an operator being unable to leave a stuck release."""
    remote = ReleaseRecordingRemote(close_delay_s=3.0)
    policy = connected_policy(monkeypatch, remote)
    threading.Timer(0.1, send_sigint).start()
    threading.Timer(0.3, send_sigint).start()

    started = time.monotonic()
    policy._atexit_close()

    assert time.monotonic() - started < 2.0
    assert "(abandoned)" in capsys.readouterr().err


@posix_only
def test_sigterm_becomes_keyboard_interrupt_only_while_a_session_is_open(monkeypatch) -> None:
    """Catch SIGTERM killing the process with no exit cleanup, or the hook outliving it."""
    assert signal.getsignal(signal.SIGTERM) is signal.SIG_DFL
    policy = connected_policy(monkeypatch, ReleaseRecordingRemote())

    assert signal.getsignal(signal.SIGTERM) is adapter_policy._raise_keyboard_interrupt
    with pytest.raises(KeyboardInterrupt, match="SIGTERM"):
        os.kill(os.getpid(), signal.SIGTERM)
        time.sleep(1.0)

    policy.close()
    assert signal.getsignal(signal.SIGTERM) is signal.SIG_DFL


@posix_only
def test_an_existing_sigterm_handler_is_never_replaced(monkeypatch) -> None:
    """Catch the adapter overriding a handler the embodiment or host installed."""

    def host_handler(_signum: int, _frame: object) -> None:
        return None

    signal.signal(signal.SIGTERM, host_handler)
    policy = connected_policy(monkeypatch, ReleaseRecordingRemote())

    assert signal.getsignal(signal.SIGTERM) is host_handler
    policy.close()
    assert signal.getsignal(signal.SIGTERM) is host_handler


def test_bind_task_lands_in_telemetry_runtime(monkeypatch) -> None:
    """Catch the task envelope from Inspect >= 0.58 being dropped."""
    from test_policy import FakeRemotePolicy, inspect_observation, step_result

    remote = FakeRemotePolicy(step_result=step_result())
    monkeypatch.setattr(
        "inspect_robots_dreamscale.policy.dreamscale.connect", lambda *_a, **_k: remote
    )
    policy = dreamscale_policy(model="dreamzero-yam")
    policy.bind_task(type("Envelope", (), {"name": "adhoc", "max_steps": 300})())
    policy.on_trial_start("s", 1, "/tmp", "run")
    policy.reset(Scene(id="s", instruction="spell NEURIPS"))
    policy.act(inspect_observation())

    runtime = policy._telemetry_rows[0]["runtime"]
    assert runtime["task"] == {"name": "adhoc", "max_steps": 300}
    assert runtime["capture_timing"] == "per_camera"
    assert runtime["capture_time_source"] == "embodiment_unix_epoch_seconds"


def test_stock_yam_observation_uses_fallback_timing_and_says_so(monkeypatch, capsys) -> None:
    """Catch stock inspect-robots-yam observations being rejected or unlabeled."""
    from inspect_robots.types import Observation
    from test_policy import FakeRemotePolicy, inspect_observation, step_result

    remote = FakeRemotePolicy(step_result=step_result())
    monkeypatch.setattr(
        "inspect_robots_dreamscale.policy.dreamscale.connect", lambda *_a, **_k: remote
    )
    policy = dreamscale_policy(model="dreamzero-yam")
    policy.on_trial_start("s", 1, "/tmp", "run")
    policy.reset(Scene(id="s", instruction="spell NEURIPS"))
    strict = inspect_observation()
    stock = Observation(images=strict.images, state=strict.state, extra=strict.extra)

    policy.act(stock)
    policy.act(stock)

    runtime = policy._telemetry_rows[0]["runtime"]
    assert runtime["capture_timing"] == "observation_fallback"
    assert runtime["capture_time_source"] == "adapter_wall_clock_at_act"
    assert capsys.readouterr().err.count("no per-camera image_times") == 1


def _process_env(home: Path) -> dict[str, str]:
    """An isolated SDK home holding a fake key, so no real config is read."""
    home.mkdir(parents=True, exist_ok=True)
    (home / "config.toml").write_text('[auth]\napi_key = "dreamscale_sk_fake"\n')
    return {
        **os.environ,
        "DREAMSCALE_HOME": str(home),
        "PYTHONPATH": str(FAKE_CONTROL_PLANE.parent),
    }


def _run_process(state: Path, home: Path, keep_warm: str) -> dict[str, Any]:
    env = _process_env(home)
    completed = subprocess.run(
        [sys.executable, str(FAKE_CONTROL_PLANE), str(state), keep_warm],
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
        check=True,
    )
    report = json.loads(completed.stdout.strip().splitlines()[-1])
    report["stderr"] = completed.stderr
    return report


@posix_only
def test_second_process_reclaims_the_session_the_first_process_parked(tmp_path) -> None:
    """Catch warm-by-default not actually carrying a session across processes.

    Two real interpreter processes share only a (fake) control plane. Neither
    calls close(); the first must park its session from the exit hook, and the
    second's create must take the reclaim path and get the same session back.
    """
    from fake_control_plane import read_state

    state = tmp_path / "control-plane.json"
    first = _run_process(state, tmp_path / "home", "default")
    second = _run_process(state, tmp_path / "home", "0")

    assert first["acquisition"] == "new"
    assert first["keep_warm_s"] == 300
    assert "warm hold of up to 300 s" in first["stderr"]
    assert second["acquisition"] == "reclaimed"
    assert second["session_id"] == first["session_id"]
    assert "reclaimed warm" in second["stderr"]
    assert "stopped session" in second["stderr"]
    events = read_state(state)["events"]
    assert events == [
        {"kind": "create", "session_id": first["session_id"], "path": "cold"},
        {"kind": "detach", "session_id": first["session_id"], "result": "parked"},
        {"kind": "create", "session_id": first["session_id"], "path": "reclaim"},
        {"kind": "delete", "session_id": first["session_id"]},
    ]


@posix_only
def test_a_process_after_the_hold_expires_starts_cold(tmp_path) -> None:
    """Catch a reclaim of a session whose warm hold already lapsed."""
    from fake_control_plane import read_state

    state = tmp_path / "control-plane.json"
    first = _run_process(state, tmp_path / "home", "1")
    time.sleep(1.2)
    second = _run_process(state, tmp_path / "home", "0")

    assert second["acquisition"] == "new"
    assert second["session_id"] != first["session_id"]
    assert [event["kind"] for event in read_state(state)["events"]] == [
        "create",
        "detach",
        "create",
        "delete",
    ]


@posix_only
def test_sigterm_mid_run_still_releases_the_session(tmp_path) -> None:
    """Catch SIGTERM (e.g. a supervisor or `kill`) leaking a session until lease expiry."""
    from fake_control_plane import read_state

    state = tmp_path / "control-plane.json"
    script = textwrap.dedent(
        f"""
        import sys, time
        sys.argv = ["fake", {str(state)!r}, "0"]
        import fake_control_plane
        fake_control_plane.main()
        print("READY", flush=True)
        time.sleep(30)
        """
    )
    env = _process_env(tmp_path / "home")
    process = subprocess.Popen(
        [sys.executable, "-c", script],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    assert process.stdout is not None
    for line in process.stdout:
        if line.strip() == "READY":
            break
    process.send_signal(signal.SIGTERM)
    _stdout, stderr = process.communicate(timeout=30)

    assert process.returncode != 0
    assert "stopped session" in stderr
    assert [event["kind"] for event in read_state(state)["events"]] == ["create", "delete"]


def test_exit_release_runs_before_executors_shut_down(tmp_path) -> None:
    """Catch the exit release running too late to open a network connection.

    Plain ``atexit`` callbacks run after ``concurrent.futures`` has shut down,
    so an SDK close that needs a fresh control-plane connection (the HTTP
    keep-alive is 5 s, the heartbeat 20 s) fails with "cannot schedule new
    futures after shutdown". This remote's close resolves a hostname on a
    pre-warmed background loop, as the SDK's httpx client would.
    """
    script = textwrap.dedent(
        """
        import asyncio, threading
        import inspect_robots_dreamscale.policy as adapter_policy
        from inspect_robots_dreamscale import dreamscale_policy

        loop = asyncio.new_event_loop()
        threading.Thread(target=loop.run_forever, daemon=True).start()

        async def resolve():
            return await asyncio.get_running_loop().getaddrinfo("localhost", 80)

        asyncio.run_coroutine_threadsafe(resolve(), loop).result(10)

        class Remote:
            session_id = "sess-exit"
            def close(self):
                asyncio.run_coroutine_threadsafe(resolve(), loop).result(10)

        adapter_policy.dreamscale.connect = lambda *_a, **_k: Remote()
        policy = dreamscale_policy(model="dreamzero-yam", keep_warm_s=0)
        policy.prepare()
        # No close(): the interpreter exit hook must release the session.
        """
    )
    completed = subprocess.run(
        [sys.executable, "-c", script],
        env=_process_env(tmp_path / "home"),
        capture_output=True,
        text=True,
        timeout=60,
    )

    assert completed.returncode == 0, completed.stderr
    assert "stopped session sess-exit" in completed.stderr
    assert "could not confirm" not in completed.stderr
