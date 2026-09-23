# inspect-robots-dropbear

This package is now [`inspect-robots-dreamscale`](https://pypi.org/project/inspect-robots-dreamscale/).
New installs should use it directly:

```bash
uv add inspect-robots-dreamscale
uv run dreamscale login
```

and select the policy with `--policy dreamscale`.

This release exists so existing setups keep working after an upgrade. It installs
`inspect-robots-dreamscale` and the `dropbear` SDK compatibility package, so all of these
continue to work unchanged:

- `--policy dropbear`, which still writes `dropbear_telemetry` and `dropbear/<run_id>/…`
  artifacts;
- `import inspect_robots_dropbear` and `from inspect_robots_dropbear.policy import DropbearPolicy`;
- `import dropbear`, the `dropbear` command, and an existing `~/.dropbear` login.
