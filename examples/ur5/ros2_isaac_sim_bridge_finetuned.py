#!/usr/bin/env python3
"""ROS2 bridge for the fine-tuned pi05_ur5 policy.

Matches the observation format used during training on gabbobosca/mir_ur_pick_place:
  - exterior_image_left  (base/left camera)
  - exterior_image_right (right exterior camera; falls back to left if not published)
  - wrist_image
  - state (7-dim: 6 joint positions + binary gripper)

60 Hz timer: when the action chunk is exhausted, calls the policy server (blocking)
and fills a new chunk. Each action is held for two physics steps → 30 fps effective
rate matching training.

Run:
    python3 examples/ur5/ros2_isaac_sim_bridge_finetuned.py \
        --host localhost --port 8000 \
        --prompt "pick up the red cube from the table"
"""

import argparse
import pathlib

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import Image, JointState

from openpi_client import websocket_client_policy as _websocket_client_policy
from openpi_client.image_tools import resize_with_pad

np.set_printoptions(precision=4, suppress=True)

LEFT_CAMERA_TOPIC = "/rgb_left"
RIGHT_CAMERA_TOPIC = "/rgb_right"
WRIST_CAMERA_TOPIC = "/rgb_wrist"
JOINT_STATE_TOPIC = "/joint_states"
ACTION_TOPIC = "joint_command"

UR5E_JOINT_NAMES = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
    "left_finger_joint",
    "right_finger_joint",
]

EXECUTION_HZ = 60  # physics rate; each action held for 2 steps → 30 fps effective


class UR5eFinetunedBridge(Node):
    def __init__(self, host: str, port: int, prompt: str,
                 left_camera_topic: str, right_camera_topic: str, wrist_camera_topic: str,
                 debug: bool = False, save_images: bool = False) -> None:
        super().__init__("ur5e_finetuned_bridge")

        self._bridge = CvBridge()
        self._prompt = prompt
        self._debug = debug
        self._save_images = save_images

        self._left_image: np.ndarray | None = None
        self._right_image: np.ndarray | None = None
        self._wrist_image: np.ndarray | None = None
        self._joint_positions: np.ndarray | None = None

        # Current action chunk (N, 7) and tick counter.
        # tick // 2 gives the action index; each action occupies 2 ticks.
        self._chunk: np.ndarray | None = None
        self._tick: int = 0

        # Binary gripper state fed to the model (1.0=open, 0.0=closed).
        # Initialized open, matching the episode-start convention in training data.
        self._gripper_state: float = 1.0

        self.create_subscription(Image, left_camera_topic, self._left_image_cb, 10)
        self.create_subscription(Image, right_camera_topic, self._right_image_cb, 10)
        self.create_subscription(Image, wrist_camera_topic, self._wrist_image_cb, 10)
        self.create_subscription(JointState, JOINT_STATE_TOPIC, self._joint_state_cb, 10)

        self._action_pub = self.create_publisher(JointState, ACTION_TOPIC, 10)

        self.get_logger().info(f"Connecting to openpi server at {host}:{port} ...")
        self._policy = _websocket_client_policy.WebsocketClientPolicy(host=host, port=port)
        self.get_logger().info(f"Connected. Metadata: {self._policy.get_server_metadata()}")

        self.create_timer(1.0 / EXECUTION_HZ, self._step)

    # ------------------------------------------------------------------
    # Subscription callbacks
    # ------------------------------------------------------------------

    def _left_image_cb(self, msg: Image) -> None:
        self._left_image = resize_with_pad(self._bridge.imgmsg_to_cv2(msg, desired_encoding="rgb8"), 224, 224)

    def _right_image_cb(self, msg: Image) -> None:
        self._right_image = resize_with_pad(self._bridge.imgmsg_to_cv2(msg, desired_encoding="rgb8"), 224, 224)

    def _wrist_image_cb(self, msg: Image) -> None:
        self._wrist_image = resize_with_pad(self._bridge.imgmsg_to_cv2(msg, desired_encoding="rgb8"), 224, 224)

    def _joint_state_cb(self, msg: JointState) -> None:
        name_to_pos = dict(zip(msg.name, msg.position))
        self._joint_positions = np.array([name_to_pos.get(n, 0.0) for n in UR5E_JOINT_NAMES], dtype=np.float32)

    # ------------------------------------------------------------------
    # Main step (60 Hz)
    # ------------------------------------------------------------------

    def _step(self) -> None:

        if self._joint_positions is None or self._left_image is None or self._wrist_image is None:
            self.get_logger().info("Waiting for sensors...", throttle_duration_sec=5.0)
            return

        if self._chunk is None or self._tick // 2 >= len(self._chunk):

            left_img  = self._left_image  if self._left_image  is not None else np.zeros((224, 224, 3), dtype=np.uint8)
            right_img = self._right_image if self._right_image is not None else left_img
            wrist_img = self._wrist_image if self._wrist_image is not None else np.zeros((224, 224, 3), dtype=np.uint8)
            joints = self._joint_positions[:6].copy()
            joints[4] = -np.pi / 2  # wrist_2 std=0 in norm_stats → snap to training mean
            state = np.concatenate([joints, [self._gripper_state]], dtype=np.float32)

            obs = {
                "exterior_image_left":  left_img,
                "exterior_image_right": right_img,
                "wrist_image":          wrist_img,
                "state":                state,
                "prompt":               self._prompt,
            }

            if self._left_image is None:
                self.get_logger().warn("Left image is None")
            else:
                self.get_logger().info(
                    f"Left image shape={self._left_image.shape}, "
                    f"min={self._left_image.min()}, "
                    f"max={self._left_image.max()}, "
                    f"mean={self._left_image.mean():.2f}"
                )

            #inference of the VLA model
            try:
                result = self._policy.infer(obs)
            except Exception as exc:
                self.get_logger().error(f"Inference error: {exc}")
                return

            actions = np.asarray(result.get("actions", []), dtype=np.float32)
            if actions.ndim == 1:
                actions = actions[np.newaxis, :]

            if self._save_images:
                pathlib.Path("debug_images").mkdir(exist_ok=True)
                cv2.imwrite("debug_images/left.png",  cv2.cvtColor(left_img,  cv2.COLOR_RGB2BGR))
                cv2.imwrite("debug_images/right.png", cv2.cvtColor(right_img, cv2.COLOR_RGB2BGR))
                cv2.imwrite("debug_images/wrist.png", cv2.cvtColor(wrist_img, cv2.COLOR_RGB2BGR))

            #debug flag
            if self._debug:
                self.get_logger().info(f"New chunk: shape={actions.shape}  actions[0]={actions[0]}  actions[-1]={actions[-1]}")

            self._chunk = actions
            self._tick = 0

        action_7 = self._chunk[self._tick // 2]
        self._tick += 1

        self._gripper_state = 1.0 if float(action_7[6]) >= 0.5 else 0.0

        # Training: 1.0=open, 0.0=closed → Isaac Sim: 0.0=open, 1.0=closed → invert.
        gripper_cmd = 1.0 - float(action_7[6])
        joint_cmd = np.concatenate([action_7[:6], [gripper_cmd, gripper_cmd]])

        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = UR5E_JOINT_NAMES
        msg.position = joint_cmd.tolist()
        self._action_pub.publish(msg)


def main() -> None:
    parser = argparse.ArgumentParser(description="UR5e fine-tuned pi05_ur5 ROS2 bridge")
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--prompt", default="pick up the red cube from the table")
    parser.add_argument("--left-camera-topic",  default=LEFT_CAMERA_TOPIC)
    parser.add_argument("--right-camera-topic", default=RIGHT_CAMERA_TOPIC)
    parser.add_argument("--wrist-camera-topic", default=WRIST_CAMERA_TOPIC)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--save-images", action="store_true", help="Save first inference images to debug_images/")
    args, ros_args = parser.parse_known_args()

    rclpy.init(args=ros_args)
    node = UR5eFinetunedBridge(
        host=args.host,
        port=args.port,
        prompt=args.prompt,
        left_camera_topic=args.left_camera_topic,
        right_camera_topic=args.right_camera_topic,
        wrist_camera_topic=args.wrist_camera_topic,
        debug=args.debug,
        save_images=args.save_images,
    )
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
