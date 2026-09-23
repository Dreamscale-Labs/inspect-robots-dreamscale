"""Startup progress: a cold start must never look frozen, on a terminal or in a log."""

from __future__ import annotations

import io
import threading
import time

import pytest

from inspect_robots_dreamscale import policy as policy_module
from inspect_robots_dreamscale.policy import _format_elapsed, _StartupProgress

LONG_STAGE = (
    "Worker allocated in us-west-2 on 2x NVIDIA B200 (modal target dreamzero-yam@us-west-2#nvfp4); "
    "loading the NVFP4 engine and warming the CFG-parallel denoiser"
)


class _Clock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


def _spinner_frames(text: str) -> list[str]:
    """Every spinner redraw, as the terminal shows it after its clear-line escape."""
    return [
        chunk.replace("\x1b[2K", "") for chunk in text.split("\r")[1:] if chunk.strip("\x1b[2K")
    ]


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
    progress = _StartupProgress(
        "starting compute", stream=stream, interactive=False, tick_s=0.01, clock=clock
    )
    with progress:
        progress.update("Worker allocated   in us-west-2")
        clock.now = 91.0
        assert beat.wait(2.0)
        progress.update("")  # blank SDK lines are ignored
    out = stream.getvalue().splitlines()
    assert out[0] == "dreamscale: starting compute"
    assert "dreamscale: Worker allocated in us-west-2" in out
    assert any(line.startswith("dreamscale: still starting … 1m31s") for line in out)
    assert "\r" not in stream.getvalue()


@pytest.mark.parametrize("columns", [12, 20, 30, 40, 60, 80, 120, 200])
def test_spinner_is_one_short_line_that_never_exceeds_the_terminal(columns: int) -> None:
    """Catch SDK stage text cluttering the spinner, or a long line wrapping (\\r can't erase it)."""
    stream = io.StringIO()
    clock = _Clock()
    progress = _StartupProgress(
        "starting DreamZero-YAM compute (a cold start can take several minutes)",
        stream=stream,
        interactive=True,
        tick_s=0.01,
        clock=clock,
        columns=lambda: columns,
    )
    with progress:
        clock.now = 125.0
        progress.update(LONG_STAGE)
        progress.update(
            "Starting your compute: 0s elapsed (estimate 900s; model load can take longer)"
        )
        deadline = time.monotonic() + 2.0
        while "2m05s" not in stream.getvalue() and time.monotonic() < deadline:
            time.sleep(0.01)
    text = stream.getvalue()
    frames = _spinner_frames(text)
    assert frames, "the spinner never drew"
    assert "\n" not in text
    assert all(len(frame) <= columns - 1 for frame in frames)
    assert not any("Worker" in frame or "estimate" in frame for frame in frames)
    # Elapsed time leads the line, so even the narrowest terminal still shows it.
    assert "2m05s" in frames[-1]
    full = "2m05s dreamscale: starting compute (a few mins)"
    if columns >= 60:
        assert frames[-1].endswith(full)
    else:
        assert frames[-1].endswith("…")


def test_spinner_follows_a_terminal_resize() -> None:
    """Catch the width being measured once, so shrinking the window wraps the line."""
    stream = io.StringIO()
    width = [120]
    clock = _Clock()
    progress = _StartupProgress(
        "starting compute",
        stream=stream,
        interactive=True,
        tick_s=60,
        clock=clock,
        columns=lambda: width[0],
    )
    with progress:
        first = _spinner_frames(stream.getvalue())[-1]
        width[0] = 20
        progress._render(1)
    last = _spinner_frames(stream.getvalue())[-1]
    assert first.endswith("starting compute (a few mins)")
    assert len(last) <= 19 and last.endswith("…")


def test_interactive_clears_line_on_exit_and_on_error() -> None:
    """Catch the spinner leaving a half-drawn line that later output would overwrite."""
    for raising in (False, True):
        stream = io.StringIO()
        progress = _StartupProgress(
            "starting compute", stream=stream, interactive=True, tick_s=0.01
        )
        if raising:
            with pytest.raises(KeyboardInterrupt), progress:
                raise KeyboardInterrupt
        else:
            with progress:
                progress.update("Transport connected: wss_tunnel")
        assert stream.getvalue().endswith("\r\x1b[2K")


def test_lines_after_startup_are_plain_not_a_revived_spinner() -> None:
    """Catch the SDK's close-time run summary redrawing the startup spinner."""
    stream = io.StringIO()
    progress = _StartupProgress(
        "starting compute", stream=stream, interactive=True, tick_s=0.01, columns=lambda: 80
    )
    with progress:
        pass
    before = stream.getvalue()
    progress.update("Run summary: transport=wss_tunnel, requests=619, chunks=67")
    after = stream.getvalue()[len(before) :]
    assert after == "dreamscale: Run summary: transport=wss_tunnel, requests=619, chunks=67\n"


def test_broken_stream_never_breaks_startup() -> None:
    """Catch a closed stderr turning a working cold start into a crash."""

    class Broken:
        def isatty(self) -> bool:
            raise OSError("closed")

        def fileno(self) -> int:
            raise OSError("closed")

        def write(self, _text: str) -> int:
            raise OSError("closed")

        def flush(self) -> None:
            raise OSError("closed")

    with _StartupProgress("starting compute", stream=Broken(), tick_s=0.01) as progress:
        progress.update("Worker allocated")
    progress.update("Run summary")


def test_elapsed_reads_naturally() -> None:
    assert [_format_elapsed(s) for s in (0, 59, 60, 106, 725)] == [
        "0s",
        "59s",
        "1m00s",
        "1m46s",
        "12m05s",
    ]
