"""Transforms for the g1_dex1_head (Unitree G1 + Dex1-1 grippers + 2-DoF head) LocoDexGrasp data.

Dataset (MagicSim converter `convert_raw_to_lerobot_g1_dex1_head.py`, LeRobot v2.1, 50 fps):
  observation.state (33):       29 body joints (ScaleBFM order), grippers [left, right] (finger position), head [yaw, pitch]
  observation.pelvis_pose (7):  pelvis world pose [x, y, z, qw, qx, qy, qz]; not a policy input
  action (40):                  pelvis world pose (7), 29 body joints, grippers [left, right] command (0/1), head [yaw, pitch]
                                = the ScaleBFM WholeBody-14 reference frame (pelvis + 29 joints) + grippers + head

What the model predicts (G1Inputs, training): the action chunk with
  - the pelvis relative to the current pelvis heading: [dx, dy] in the heading frame (yaw only), z absolute (height above
    the floor), orientation q_yaw0^-1 * q (relative yaw, absolute roll/pitch, both gravity aligned)
  - joints and head as deltas to the current state (as openpi does for absolute joint actions); grippers absolute
G1Outputs (inference) adds the current joints/head back. The pelvis stays heading-relative: the robot side composes it with
the current pelvis pose (world = p0 + Rz(yaw0) [dx, dy], z; q = q_yaw0 * q_rel) to build the ScaleBFM reference.
"""

import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model

STATE_DIM = 33
ACTION_DIM = 40
PELVIS = slice(0, 7)
JOINTS_A, JOINTS_S = slice(7, 36), slice(0, 29)  # body joints in action / state
GRIPPERS_A = slice(36, 38)
HEAD_A, HEAD_S = slice(38, 40), slice(31, 33)


def make_g1_example() -> dict:
    """A random input example as the robot sends it (inference)."""
    return {
        "observation/state": np.random.rand(STATE_DIM).astype(np.float32),
        "observation/head": np.random.randint(256, size=(480, 640, 3), dtype=np.uint8),
        "observation/left_wrist": np.random.randint(256, size=(480, 640, 3), dtype=np.uint8),
        "observation/right_wrist": np.random.randint(256, size=(480, 640, 3), dtype=np.uint8),
        "prompt": "Pick up the apple.",
    }


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


def _quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Hamilton product of wxyz quaternions, broadcast over leading dims."""
    w1, x1, y1, z1 = np.moveaxis(a, -1, 0)
    w2, x2, y2, z2 = np.moveaxis(b, -1, 0)
    return np.stack([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                     w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2], axis=-1)


def _yaw(q: np.ndarray) -> np.ndarray:
    w, x, y, z = np.moveaxis(q, -1, 0)
    return np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def pelvis_to_heading_frame(pelvis_pose: np.ndarray, pelvis_chunk: np.ndarray) -> np.ndarray:
    """World pelvis poses (H, 7) -> relative to the heading of the current pelvis pose (7,). See the module docstring."""
    yaw0 = _yaw(pelvis_pose[3:7])
    c, s = np.cos(yaw0), np.sin(yaw0)
    d = pelvis_chunk[:, :2] - pelvis_pose[:2]
    xy = np.stack([c * d[:, 0] + s * d[:, 1], -s * d[:, 0] + c * d[:, 1]], axis=-1)  # Rz(-yaw0) d
    q_yaw0_inv = np.array([np.cos(yaw0 / 2), 0.0, 0.0, -np.sin(yaw0 / 2)])
    q = _quat_mul(q_yaw0_inv, pelvis_chunk[:, 3:7])
    q *= np.where(q[:, :1] < 0, -1.0, 1.0)  # w >= 0
    return np.concatenate([xy, pelvis_chunk[:, 2:3], q], axis=-1)


@dataclasses.dataclass(frozen=True)
class G1Inputs(transforms.DataTransformFn):
    """Dataset / robot dict -> model inputs. Used for training and inference."""

    model_type: _model.ModelType
    # Joints and head as deltas to the current state (grippers stay absolute).
    delta_joints: bool = True

    def __call__(self, data: dict) -> dict:
        state = np.asarray(data["observation/state"], dtype=np.float32)
        inputs = {
            "state": state,
            "image": {
                "base_0_rgb": _parse_image(data["observation/head"]),
                "left_wrist_0_rgb": _parse_image(data["observation/left_wrist"]),
                "right_wrist_0_rgb": _parse_image(data["observation/right_wrist"]),
            },
            "image_mask": {"base_0_rgb": np.True_, "left_wrist_0_rgb": np.True_, "right_wrist_0_rgb": np.True_},
        }

        # Actions only exist in training.
        if "actions" in data:
            actions = np.array(data["actions"], dtype=np.float32)  # (H, 40), copy
            actions[:, PELVIS] = pelvis_to_heading_frame(
                np.asarray(data["observation/pelvis_pose"], dtype=np.float64), actions[:, PELVIS].astype(np.float64)
            )
            if self.delta_joints:
                actions[:, JOINTS_A] -= state[JOINTS_S]
                actions[:, HEAD_A] -= state[HEAD_S]
            inputs["actions"] = actions

        if "prompt" in data:
            inputs["prompt"] = data["prompt"]
        return inputs


@dataclasses.dataclass(frozen=True)
class G1Outputs(transforms.DataTransformFn):
    """Model outputs -> robot actions (inference): joints and head absolute again, pelvis heading-relative."""

    delta_joints: bool = True

    def __call__(self, data: dict) -> dict:
        actions = np.array(data["actions"][..., :ACTION_DIM])
        if self.delta_joints:
            state = np.asarray(data["state"])[..., :STATE_DIM]
            actions[..., JOINTS_A] += state[..., None, JOINTS_S]
            actions[..., HEAD_A] += state[..., None, HEAD_S]
        return {"actions": actions}
