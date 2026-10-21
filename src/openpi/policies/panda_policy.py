"""Data transforms for Franka Panda datasets recorded in Isaac Sim with ter_grasp.

Dataset (LeRobot, written by ter_grasp/convert_to_lerobot.py, 30 fps):
  exterior_image_left  : back-left exterior camera
  exterior_image_right : back-right exterior camera
  wrist_image          : camera on the Franka Hand
  state                : [q1..q7 (rad), gripper]  -- q = commanded or measured joint positions
  actions              : [q1..q7 (rad), gripper]  -- absolute, next-frame commanded joint targets

The gripper channel is *closedness* (0 = open, 1 = closed) in the dataset, in the model and
on the robot: the same convention as the pi pre-training data (DROID, Franka), so it passes
through unchanged.  Datasets converted before ter_grasp switched to this convention store
1 = open and must be re-converted.
"""

import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model

# 7 arm joints + 1 gripper channel.
ACTION_DIM = 8


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class PandaInputs(transforms.DataTransformFn):
    model_type: _model.ModelType = _model.ModelType.PI0

    def __call__(self, data: dict) -> dict:
        left_image = _parse_image(data["exterior_image_left"])
        right_image = _parse_image(data["exterior_image_right"])
        wrist_image = _parse_image(data["wrist_image"])

        inputs = {
            "state": np.asarray(data["state"]),
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
            inputs["actions"] = np.asarray(data["actions"])

        if "prompt" in data:
            prompt = data["prompt"]
            if isinstance(prompt, bytes):
                prompt = prompt.decode("utf-8")
            inputs["prompt"] = prompt

        return inputs


@dataclasses.dataclass(frozen=True)
class PandaOutputs(transforms.DataTransformFn):
    def __call__(self, data: dict) -> dict:
        # The model emits 32-dim padded actions; keep the first 8. `...` keeps this valid for
        # batched outputs too.
        return {"actions": np.asarray(data["actions"])[..., :ACTION_DIM]}
