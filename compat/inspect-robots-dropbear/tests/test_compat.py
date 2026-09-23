from importlib.metadata import entry_points


def test_old_import_paths_resolve_to_the_renamed_modules() -> None:
    """Catch the compatibility package drifting from the adapter it re-exports."""
    import inspect_robots_dreamscale
    import inspect_robots_dreamscale.policy
    import inspect_robots_dropbear
    from inspect_robots_dropbear.policy import DropbearPolicy

    assert inspect_robots_dropbear.DropbearPolicy is inspect_robots_dreamscale.DropbearPolicy
    assert DropbearPolicy is inspect_robots_dreamscale.policy.DropbearPolicy
    assert inspect_robots_dropbear.dropbear_policy(model="dreamzero-yam").info.name == "dropbear"


def test_the_dropbear_policy_name_is_registered_once_by_the_adapter() -> None:
    """Catch a duplicate `dropbear` entry point racing the adapter's alias."""
    names = [ep for ep in entry_points(group="inspect_robots.policies") if ep.name == "dropbear"]

    assert [ep.value for ep in names] == ["inspect_robots_dreamscale:dropbear_policy"]


def test_the_sdk_compatibility_package_is_installed() -> None:
    """Catch an upgrade that silently removes `import dropbear` for existing scripts."""
    import dreamscale
    import dropbear

    assert dropbear.__version__ == dreamscale.__version__
