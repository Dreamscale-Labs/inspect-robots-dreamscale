import re

import numpy as np
import pytest
from inspect_robots.types import Observation

from inspect_robots_dreamscale.dreamzero_yam import to_dreamzero_yam

CAPTURE_EPOCH_S = 1_700_000_000.0


@pytest.fixture(autouse=True)
def fixed_wall_clock(monkeypatch) -> None:
    monkeypatch.setattr(
        "inspect_robots_dreamscale.dreamzero_yam.time.time",
        lambda: CAPTURE_EPOCH_S + 0.010,
    )


class RefusesArrayConversion:
    def __array__(self, *args: object, **kwargs: object) -> np.ndarray:
        del args, kwargs
        raise RuntimeError("custom conversion detail")


def frame(value: int) -> np.ndarray:
    return np.full((2, 3, 3), value, dtype=np.uint8)


def valid_observation() -> Observation:
    return Observation(
        images={
            "top_cam": frame(1),
            "left_cam": frame(2),
            "right_cam": frame(3),
        },
        state={"joint_pos": np.arange(14, dtype=np.float64)},
        instruction="spell NEURIPS",
        image_times={
            "top_cam": CAPTURE_EPOCH_S,
            "left_cam": CAPTURE_EPOCH_S + 0.001,
            "right_cam": CAPTURE_EPOCH_S + 0.002,
        },
        extra={"env_step": 8},
    )


def test_to_dreamzero_yam_maps_cameras_times_and_packed_state() -> None:
    """Catch a camera, timestamp, or joint-order mapping mismatch at the SDK boundary."""
    mapped = to_dreamzero_yam(valid_observation())

    assert mapped.camera_capture_times_ns == (
        round(CAPTURE_EPOCH_S * 1_000_000_000),
        round((CAPTURE_EPOCH_S + 0.001) * 1_000_000_000),
        round((CAPTURE_EPOCH_S + 0.002) * 1_000_000_000),
    )
    assert mapped.left_joint_positions == (0.0, 1.0, 2.0, 3.0, 4.0, 5.0)
    assert mapped.left_gripper == 6.0
    assert mapped.right_joint_positions == (7.0, 8.0, 9.0, 10.0, 11.0, 12.0)
    assert mapped.right_gripper == 13.0


@pytest.mark.parametrize(
    ("observation", "message"),
    [
        (
            Observation(
                images={"left_cam": frame(2), "right_cam": frame(3)},
                state={"joint_pos": np.arange(14, dtype=np.float64)},
                image_times={
                    "top_cam": CAPTURE_EPOCH_S,
                    "left_cam": CAPTURE_EPOCH_S + 0.001,
                    "right_cam": CAPTURE_EPOCH_S + 0.002,
                },
            ),
            "missing images['top_cam']",
        ),
        (
            Observation(
                images={"top_cam": frame(1), "left_cam": frame(2), "right_cam": frame(3)},
                state={"joint_pos": np.arange(14, dtype=np.float64)},
                image_times={
                    "left_cam": CAPTURE_EPOCH_S + 0.001,
                    "right_cam": CAPTURE_EPOCH_S + 0.002,
                },
            ),
            "missing image_times['top_cam']",
        ),
        (
            Observation(
                images={"top_cam": frame(1), "left_cam": frame(2), "right_cam": frame(3)},
                state={},
                image_times={
                    "top_cam": CAPTURE_EPOCH_S,
                    "left_cam": CAPTURE_EPOCH_S + 0.001,
                    "right_cam": CAPTURE_EPOCH_S + 0.002,
                },
            ),
            "missing state['joint_pos']",
        ),
    ],
)
def test_to_dreamzero_yam_names_each_missing_required_observation_key(
    observation: Observation, message: str
) -> None:
    """Catch a KeyError or an ambiguous error when an Inspect mapping is incomplete."""
    with pytest.raises(ValueError, match=re.escape(message)):
        to_dreamzero_yam(observation)


@pytest.mark.parametrize("value", [True, np.bool_(True), -0.1, float("nan"), float("inf")])
def test_to_dreamzero_yam_rejects_invalid_camera_time(value: object) -> None:
    """Catch malformed wall-clock seconds before they are converted to nanoseconds."""
    observation = valid_observation()
    observation = Observation(
        images=observation.images,
        state=observation.state,
        image_times={**observation.image_times, "top_cam": value},
    )

    with pytest.raises(
        ValueError, match=r"image_times\['top_cam'\] must be finite Unix-epoch seconds"
    ):
        to_dreamzero_yam(observation)


def test_to_dreamzero_yam_accepts_cross_camera_skew_without_rewriting_times() -> None:
    """Free-running camera spread is observable metadata, not an adapter rejection gate."""
    observation = valid_observation()
    observation = Observation(
        images=observation.images,
        state=observation.state,
        image_times={
            "top_cam": CAPTURE_EPOCH_S,
            "left_cam": CAPTURE_EPOCH_S + 0.001,
            "right_cam": CAPTURE_EPOCH_S + 0.200,
        },
    )

    mapped = to_dreamzero_yam(observation)

    assert mapped.camera_capture_times_ns == (
        round(CAPTURE_EPOCH_S * 1_000_000_000),
        round((CAPTURE_EPOCH_S + 0.001) * 1_000_000_000),
        round((CAPTURE_EPOCH_S + 0.200) * 1_000_000_000),
    )


def test_to_dreamzero_yam_rejects_stale_camera_time() -> None:
    observation = valid_observation()
    observation = Observation(
        images=observation.images,
        state=observation.state,
        image_times={
            **observation.image_times,
            "top_cam": CAPTURE_EPOCH_S - 5.001,
        },
    )

    with pytest.raises(ValueError, match=r"image_times\['top_cam'\] is stale"):
        to_dreamzero_yam(observation)


def test_to_dreamzero_yam_rejects_implausibly_future_camera_time() -> None:
    observation = valid_observation()
    observation = Observation(
        images=observation.images,
        state=observation.state,
        image_times={
            **observation.image_times,
            "top_cam": CAPTURE_EPOCH_S + 1.011,
        },
    )

    with pytest.raises(ValueError, match=r"image_times\['top_cam'\] is implausibly future"):
        to_dreamzero_yam(observation)


@pytest.mark.parametrize(
    "joint_pos",
    [
        np.arange(13, dtype=np.float64),
        np.array([True] * 14),
        np.array([0.0] * 13 + [float("nan")]),
    ],
)
def test_to_dreamzero_yam_rejects_invalid_joint_positions(joint_pos: np.ndarray) -> None:
    """Catch malformed YAM state before joint values are packed by arm and gripper."""
    observation = valid_observation()
    observation = Observation(
        images=observation.images,
        state={"joint_pos": joint_pos},
        image_times=observation.image_times,
    )

    with pytest.raises(ValueError, match="joint_pos must contain exactly 14 finite values"):
        to_dreamzero_yam(observation)


@pytest.mark.parametrize(
    "joint_pos",
    [
        [np.zeros((2, 2)), np.zeros((2, 3))],
        RefusesArrayConversion(),
    ],
)
def test_to_dreamzero_yam_normalizes_joint_conversion_failures(joint_pos: object) -> None:
    """Catch NumPy or custom conversion details escaping the adapter validation boundary."""
    observation = valid_observation()
    observation = Observation(
        images=observation.images,
        state={"joint_pos": joint_pos},
        image_times=observation.image_times,
    )

    with pytest.raises(
        ValueError, match=r"^joint_pos must contain exactly 14 finite values$"
    ):
        to_dreamzero_yam(observation)
