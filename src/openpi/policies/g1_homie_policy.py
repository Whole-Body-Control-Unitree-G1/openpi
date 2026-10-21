"""Transforms for the g1_dex1_head LocoDexGrasp data with HOMIE base commands as the action.

Dataset (MagicSim converter `convert_raw_to_lerobot_g1_dex1_head.py --action homie`, LeRobot v2.1, 50 fps):
  observation.state (39):  29 body joints (ScaleBFM order), grippers [left, right] (finger position), head [yaw, pitch],
                           base velocity [vx, vy, vz, wx, wy, wz] (pelvis body frame)
  action (28):             HOMIE command [vx, vy, wz, base height, torso roll, pitch, yaw], 17 waist + arm joint targets
                           (Pink IK, UPPER_JOINTS order), grippers [left, right] command (0/1), head [yaw, pitch]

Model state (36): the 33 joint / gripper / head values and the base angular velocity. The base linear velocity is
dropped: the real G1 has no measurement of it (rt/lowstate gives the IMU's angular velocity only).
Model actions: HOMIE commands and grippers absolute; waist / arm joints and head as deltas to the current state.
G1HomieOutputs (inference) adds the current joints / head back.
"""

import dataclasses

import numpy as np

from openpi import transforms
from openpi.models import model as _model
from openpi.policies.g1_policy import _parse_image

# ScaleBFM joint order of observation.state[:29] (convert_raw_to_lerobot_g1_dex1_head.SCALEBFM_JOINTS)
SCALEBFM_JOINTS = [
    "left_hip_pitch_joint", "right_hip_pitch_joint", "waist_yaw_joint", "left_hip_roll_joint", "right_hip_roll_joint",
    "waist_roll_joint", "left_hip_yaw_joint", "right_hip_yaw_joint", "waist_pitch_joint", "left_knee_joint",
    "right_knee_joint", "left_shoulder_pitch_joint", "right_shoulder_pitch_joint", "left_ankle_pitch_joint",
    "right_ankle_pitch_joint", "left_shoulder_roll_joint", "right_shoulder_roll_joint", "left_ankle_roll_joint",
    "right_ankle_roll_joint", "left_shoulder_yaw_joint", "right_shoulder_yaw_joint", "left_elbow_joint",
    "right_elbow_joint", "left_wrist_roll_joint", "right_wrist_roll_joint", "left_wrist_pitch_joint",
    "right_wrist_pitch_joint", "left_wrist_yaw_joint", "right_wrist_yaw_joint",
]  # fmt: skip
UPPER_JOINTS = [n for n in SCALEBFM_JOINTS if n.startswith(("waist", "left_shoulder", "right_shoulder", "left_elbow",
                                                            "right_elbow", "left_wrist", "right_wrist"))]  # fmt: skip
UPPER_IN_STATE = np.array([SCALEBFM_JOINTS.index(n) for n in UPPER_JOINTS])

DATASET_STATE_DIM = 39
STATE_DIM = 36
ACTION_DIM = 28
CMD_A = slice(0, 7)
UPPER_A = slice(7, 24)
GRIPPERS_A = slice(24, 26)
HEAD_A, HEAD_S = slice(26, 28), slice(31, 33)


def model_state(dataset_state: np.ndarray) -> np.ndarray:
    """39-D dataset / robot state -> 36-D model state (drop the base linear velocity)."""
    return np.concatenate([dataset_state[..., :33], dataset_state[..., 36:39]], axis=-1)


def make_g1_homie_example() -> dict:
    """A random input example as the robot sends it (inference): the 39-D state, base linear velocity may be zero."""
    return {
        "observation/state": np.random.rand(DATASET_STATE_DIM).astype(np.float32),
        "observation/head": np.random.randint(256, size=(480, 640, 3), dtype=np.uint8),
        "observation/left_wrist": np.random.randint(256, size=(480, 640, 3), dtype=np.uint8),
        "observation/right_wrist": np.random.randint(256, size=(480, 640, 3), dtype=np.uint8),
        "prompt": "Pick up the apple.",
    }


@dataclasses.dataclass(frozen=True)
class G1HomieInputs(transforms.DataTransformFn):
    """Dataset / robot dict -> model inputs. Used for training and inference."""

    model_type: _model.ModelType
    # Waist / arm joints and head as deltas to the current state (commands and grippers stay absolute).
    delta_joints: bool = True

    def __call__(self, data: dict) -> dict:
        state = model_state(np.asarray(data["observation/state"], dtype=np.float32))
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
            actions = np.array(data["actions"], dtype=np.float32)  # (H, 28), copy
            if self.delta_joints:
                actions[:, UPPER_A] -= state[UPPER_IN_STATE]
                actions[:, HEAD_A] -= state[HEAD_S]
            inputs["actions"] = actions

        if "prompt" in data:
            inputs["prompt"] = data["prompt"]
        return inputs


@dataclasses.dataclass(frozen=True)
class G1HomieOutputs(transforms.DataTransformFn):
    """Model outputs -> robot actions (inference): waist / arm joints and head absolute again."""

    delta_joints: bool = True

    def __call__(self, data: dict) -> dict:
        actions = np.array(data["actions"][..., :ACTION_DIM])
        if self.delta_joints:
            state = np.asarray(data["state"])[..., :STATE_DIM]
            actions[..., UPPER_A] += state[..., None, UPPER_IN_STATE]
            actions[..., HEAD_A] += state[..., None, HEAD_S]
        return {"actions": actions}
