"""A file-backed fake Dreamscale control plane, shared by separate test processes.

The only thing two Inspect Robots processes share is the control plane, so the
fake keeps its whole state in one JSON file. Each process gets its own client
(as the real SDK does), talks to the same "server", and runs the pinned SDK's
real ``open_policy_connection`` / ``PolicyConnection.close`` code: session
create, readiness polling, heartbeat, detach-to-park and delete. Only the
network edges -- the HTTP client, UDP preflight and the QUIC transport -- are
replaced.

The park/reclaim rules mirror ``dropbear_control_plane.sessions_api``: a detach
with ``keep_warm > 0`` parks the session until ``now + keep_warm``; a later create
in the same scope (model, region mode) claims it before allocating, returning
the same session id with ``selection_reason = "keep_warm_reuse"``.

Run as a script, it is one "process": open (or reclaim) a session through the
adapter, report it as JSON, and exit **without** calling ``close()`` -- exactly
as ``inspect-robots run`` does -- so release relies on the adapter's exit hook.
"""

from __future__ import annotations

import asyncio
import fcntl
import json
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from dreamscale.control import Session


@contextmanager
def _locked_state(path: Path):
    path.touch(exist_ok=True)
    with path.open("r+", encoding="utf-8") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        raw = handle.read()
        state = json.loads(raw) if raw else {"next_id": 1, "sessions": {}, "events": []}
        yield state
        handle.seek(0)
        handle.truncate()
        handle.write(json.dumps(state, sort_keys=True))
        handle.flush()


def read_state(path: Path) -> dict[str, Any]:
    with _locked_state(path) as state:
        return json.loads(json.dumps(state))


def _session(record: dict[str, Any]) -> Session:
    return Session(
        session_id=record["session_id"],
        status=record["status"],
        model=record["model"],
        region="us-west-2",
        target_key="dreamzero-yam@us-west-2#nvfp4",
        worker_addr="worker.invalid",
        port=4433,
        cert_fingerprint="ab" * 32,
        session_token="token",
        lease_ttl_s=60,
        region_mode=record["region_mode"],
        selection_reason=record["selection_reason"],
    )


class FakeControl:
    """One process's control-plane client, bound to the shared state file."""

    def __init__(self, state_path: Path) -> None:
        self._path = state_path

    def _event(self, state: dict[str, Any], kind: str, session_id: str, **extra: Any) -> None:
        state["events"].append({"kind": kind, "session_id": session_id, **extra})

    async def create_session(
        self,
        model: str,
        preferred_region: str | None = None,
        *,
        region_mode: str | None = None,
        keep_warm: int = 0,
        **_kwargs: Any,
    ) -> Session:
        del preferred_region
        now = time.time()
        with _locked_state(self._path) as state:
            for record in state["sessions"].values():
                if (
                    record["status"] == "parked"
                    and record["model"] == model
                    and record["region_mode"] == region_mode
                    and record["park_expires_at"] > now
                ):
                    record.update(
                        status="resuming",
                        keep_warm=keep_warm,
                        selection_reason="keep_warm_reuse",
                        park_expires_at=None,
                    )
                    self._event(state, "create", record["session_id"], path="reclaim")
                    return _session(record)
            session_id = f"sess-{state['next_id']}"
            state["next_id"] += 1
            record = {
                "session_id": session_id,
                "status": "starting",
                "model": model,
                "region_mode": region_mode,
                "keep_warm": keep_warm,
                "selection_reason": "cold_allocation",
                "park_expires_at": None,
            }
            state["sessions"][session_id] = record
            self._event(state, "create", session_id, path="cold")
            return _session(record)

    async def get_session(self, session_id: str) -> Session:
        with _locked_state(self._path) as state:
            record = state["sessions"][session_id]
            if record["status"] in {"starting", "resuming"}:
                record["status"] = "ready"
            return _session(record)

    async def heartbeat(self, _session_id: str) -> None:
        return None

    async def detach_session(self, session_id: str) -> None:
        with _locked_state(self._path) as state:
            record = state["sessions"][session_id]
            if record["keep_warm"] > 0 and record["status"] == "ready":
                record["status"] = "parked"
                record["park_expires_at"] = time.time() + record["keep_warm"]
                self._event(state, "detach", session_id, result="parked")
            else:
                record["status"] = "stopped"
                self._event(state, "detach", session_id, result="stopped")

    async def delete_session(self, session_id: str) -> None:
        with _locked_state(self._path) as state:
            state["sessions"][session_id]["status"] = "stopped"
            self._event(state, "delete", session_id)

    async def report_transport(self, _session_id: str, **_metadata: Any) -> None:
        return None

    async def close(self) -> None:
        return None


class FakeTransport:
    mode = "quic"
    fallback_reason = None
    primary_preflight_status = None
    diagnostic_preflight_status = None

    def __init__(self) -> None:
        self._queue: asyncio.Queue[Any] = asyncio.Queue()

    async def send_observation(self, _observation: Any) -> None:
        return None

    def action_chunks(self) -> Any:
        return self._iterate()

    async def _iterate(self) -> Any:
        while True:
            yield await self._queue.get()

    async def close(self) -> None:
        return None


class _UdpOk:
    ok = True
    reason = "ok"
    remediation = ""
    diagnostic = None

    class worker:  # noqa: N801 - mirrors the SDK report shape
        status = "ok"
        remediation = ""


def fake_connect(state_path: Path):
    """A drop-in for ``dreamscale.connect`` that runs the real SDK lifecycle."""
    from dreamscale.policy.lifecycle import _LoopRunner
    from dreamscale.policy.remote import RemotePolicy, _aconnect

    async def preflight(*_args: Any, **_kwargs: Any) -> _UdpOk:
        return _UdpOk()

    async def transport_connect(*_args: Any, **_kwargs: Any) -> FakeTransport:
        return FakeTransport()

    async def quick_sleep(seconds: float) -> None:
        await asyncio.sleep(min(seconds, 0.02))

    def connect(model: str, **kwargs: Any) -> RemotePolicy:
        runner = _LoopRunner()
        try:
            async_policy = runner.run(
                _aconnect(
                    model,
                    **kwargs,
                    client_factory=lambda _url, _key: FakeControl(state_path),
                    transport_connect=transport_connect,
                    preflight_fn=preflight,
                    sleep=quick_sleep,
                )
            )
        except BaseException:
            runner.shutdown()
            raise
        return RemotePolicy(runner=runner, async_policy=async_policy)

    return connect


def main() -> None:
    state_path = Path(sys.argv[1])
    keep_warm_args = {} if sys.argv[2] == "default" else {"keep_warm_s": int(sys.argv[2])}

    import inspect_robots_dreamscale.policy as adapter_policy
    from inspect_robots_dreamscale import dreamscale_policy

    adapter_policy.dreamscale.connect = fake_connect(state_path)
    policy = dreamscale_policy(model="dreamzero-yam", **keep_warm_args)
    policy.prepare()
    print(
        json.dumps(
            {
                "session_id": policy.session_id,
                "acquisition": policy._session_acquisition,
                "keep_warm_s": policy.keep_warm_s,
            }
        ),
        flush=True,
    )
    # Deliberately no close(): Inspect Robots never closes a policy.


if __name__ == "__main__":
    main()
