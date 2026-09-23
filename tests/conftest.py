"""Keep process-global side effects of the policy out of the test process."""

from __future__ import annotations

import atexit
import signal

import pytest

import inspect_robots_dreamscale.policy as adapter_policy

_SIGNALS = tuple(
    getattr(signal, name) for name in ("SIGTERM", "SIGHUP") if hasattr(signal, name)
)


@pytest.fixture(autouse=True)
def isolate_process_hooks(monkeypatch):
    """Stop exit hooks and signal handlers from leaking between tests.

    Every constructed policy registers interpreter-exit hooks and, once
    connected, termination-signal handlers. Real-exit behaviour is covered by
    the subprocess tests; in-process tests must not leave hooks that fire when
    pytest itself exits.
    """
    saved = {signum: signal.getsignal(signum) for signum in _SIGNALS}
    monkeypatch.setattr(adapter_policy, "_register_threading_exit", lambda _handler: True)
    monkeypatch.setattr(atexit, "register", lambda handler: handler)
    monkeypatch.setattr(atexit, "unregister", lambda _handler: None)
    yield
    for signum, handler in saved.items():
        signal.signal(signum, handler)
    adapter_policy._signal_owners.clear()
    adapter_policy._installed_signals.clear()
