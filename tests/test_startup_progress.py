"""Startup progress: a cold start must never look frozen, on a terminal or in a log."""

from __future__ import annotations

import io
import threading

import pytest

from inspect_robots_dreamscale import policy as policy_module
from inspect_robots_dreamscale.policy import _StartupProgress


class _Clock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


def test_non_interactive_prints_message_stages_and_heartbeat(monkeypatch) -> None:
    """Catch piped/logged runs going silent for minutes during a cold start."""
    monkeypatch.setattr(policy_module, "_HEARTBEAT_S", 30.0)
    stream = io.StringIO()
    clock = _Clock()
    beat = threading.Event()
    original_write = stream.write

    def write(text: str) -> int:
        if "still starting" in text:
            beat.set()
        return original_write(text)

    stream.write = write  # type: ignore[method-assign]
    with _StartupProgress(
        "starting compute", stream=stream, interactive=False, tick_s=0.01, clock=clock
    ) as progress:
        progress.update("Worker allocated   in us-west-2")
        clock.now = 31.0
        assert beat.wait(2.0)
        progress.update("")  # blank SDK lines are ignored
    out = stream.getvalue().splitlines()
    assert out[0] == "dreamscale: starting compute"
    assert "dreamscale: Worker allocated in us-west-2" in out
    assert any(line.startswith("dreamscale: still starting … 31s") for line in out)
    assert "\r" not in stream.getvalue()


def test_interactive_redraws_one_line_with_elapsed_and_stage_then_clears() -> None:
    """Catch the spinner leaving a half-drawn line that later output would overwrite."""
    stream = io.StringIO()
    clock = _Clock()
    with _StartupProgress(
        "starting compute", stream=stream, interactive=True, tick_s=0.01, clock=clock
    ) as progress:
        clock.now = 7.0
        progress.update("Transport connected: wss_tunnel")
    text = stream.getvalue()
    assert "dreamscale: starting compute — 0s" in text
    assert "starting compute — 7s · Transport connected: wss_tunnel" in text
    assert "\n" not in text
    assert text.endswith("\r\x1b[2K")


def test_progress_clears_line_even_when_connect_raises() -> None:
    """Catch an interrupted cold start leaving the terminal line dirty."""
    stream = io.StringIO()
    with pytest.raises(KeyboardInterrupt):
        with _StartupProgress("starting compute", stream=stream, interactive=True, tick_s=0.01):
            raise KeyboardInterrupt
    assert stream.getvalue().endswith("\r\x1b[2K")


def test_broken_stream_never_breaks_startup() -> None:
    """Catch a closed stderr turning a working cold start into a crash."""

    class Broken:
        def isatty(self) -> bool:
            raise OSError("closed")

        def write(self, _text: str) -> int:
            raise OSError("closed")

        def flush(self) -> None:
            raise OSError("closed")

    with _StartupProgress("starting compute", stream=Broken(), tick_s=0.01) as progress:
        progress.update("Worker allocated")
