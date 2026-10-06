"""Data transforms for Franka Panda datasets recorded in Isaac Sim with ter_grasp.

Dataset (LeRobot, written by ter_grasp/convert_to_lerobot.py, 30 fps):
  exterior_image_left  : back-left exterior camera
  exterior_image_right : back-right exterior camera
  wrist_image          : camera on the Franka Hand
  state                : [q1..q7 (rad), gripper]  -- q = commanded joint drive targets
  actions              : [q1..q7 (rad), gripper]  -- absolute, actions[t] = state[t + 1]

ter_grasp stores the gripper as 1.0 = open / 0.0 = closed.  The pi pre-training data
(DROID, Franka) uses gripper *closedness* (0 = open, 1 = closed), so the gripper channel
is flipped on the way in and flipped back on the way out.  The model never sees the
ter_grasp convention; the robot never sees the model's.
"""

import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model

# 7 arm joints + 1 gripper channel.
ACTION_DIM = 8
GRIPPER_IDX = 7


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


def _flip_gripper(x) -> np.ndarray:
    """open (1) <-> closed (0) on the last axis' gripper channel. Works for (8,) and (..., H, 8)."""
    x = np.array(x, dtype=np.float32, copy=True)
    x[..., GRIPPER_IDX] = 1.0 - x[..., GRIPPER_IDX]
    return x


@dataclasses.dataclass(frozen=True)
class PandaInputs(transforms.DataTransformFn):
    model_type: _model.ModelType = _model.ModelType.PI0

    def __call__(self, data: dict) -> dict:
        left_image = _parse_image(data["exterior_image_left"])
        right_image = _parse_image(data["exterior_image_right"])
        wrist_image = _parse_image(data["wrist_image"])

        inputs = {
            "state": _flip_gripper(data["state"]),
            "image": {
                "base_0_rgb": left_image,
                "left_wrist_0_rgb": wrist_image,
                "right_wrist_0_rgb": right_image,
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                "right_wrist_0_rgb": np.True_,
            },
        }

        if "actions" in data:
            inputs["actions"] = _flip_gripper(data["actions"])

        if "prompt" in data:
            prompt = data["prompt"]
            if isinstance(prompt, bytes):
                prompt = prompt.decode("utf-8")
            inputs["prompt"] = prompt

        return inputs


@dataclasses.dataclass(frozen=True)
class PandaOutputs(transforms.DataTransformFn):
    def __call__(self, data: dict) -> dict:
        # The model emits 32-dim padded actions; keep the first 8 and restore ter_grasp's
        # gripper convention (1 = open). `...` keeps this valid for batched outputs too.
        return {"actions": _flip_gripper(np.asarray(data["actions"])[..., :ACTION_DIM])}
