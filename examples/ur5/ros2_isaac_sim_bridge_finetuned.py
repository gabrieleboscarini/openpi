#!/usr/bin/env python3
"""ROS2 bridge for the fine-tuned pi05_ur5 policy.

Matches the observation format used during training on gabbobosca/mir_ur_pick_place:
  - exterior_image_left  (base/left camera)
  - exterior_image_right (right exterior camera; falls back to left if not published)
  - wrist_image
  - state (7-dim: 6 joint positions + gripper)

The policy server must be started with:
    --policy:checkpoint.config pi05_ur5
    --policy:checkpoint.dir <path_to_checkpoint/3000>

Run:
    python3 examples/ur5/ros2_isaac_sim_bridge_finetuned.py \
        --host localhost --port 8000 \
        --prompt "pick up the object"
"""

import argparse
import threading

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import Image, JointState

from openpi_client import websocket_client_policy as _websocket_client_policy

# ---------------------------------------------------------------------------
# Topic names — adjust to match your Isaac Sim ROS2 bridge configuration.
# ---------------------------------------------------------------------------
LEFT_CAMERA_TOPIC = "/camera/color/image_raw"
RIGHT_CAMERA_TOPIC = "/right_camera/color/image_raw"
WRIST_CAMERA_TOPIC = "/wrist_camera/color/image_raw"
JOINT_STATE_TOPIC = "/joint_states"
ACTION_TOPIC = "joint_command"

# UR5e joint names in the order they appear in /joint_states.
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

# Inference rate (Hz).
INFERENCE_HZ = 10


class UR5eFinetunedBridge(Node):
    def __init__(self, host: str, port: int, prompt: str, right_camera_topic: str) -> None:
        super().__init__("ur5e_finetuned_bridge")

        self._bridge = CvBridge()
        self._prompt = prompt
        self._lock = threading.Lock()

        self._left_image: np.ndarray | None = None
        self._right_image: np.ndarray | None = None
        self._wrist_image: np.ndarray | None = None
        # 8-dim: [shoulder_pan, shoulder_lift, elbow, wrist_1, wrist_2, wrist_3, left_finger, right_finger]
        self._joint_positions: np.ndarray | None = None

        self.create_subscription(Image, LEFT_CAMERA_TOPIC, self._left_image_cb, 10)
        self.create_subscription(Image, right_camera_topic, self._right_image_cb, 10)
        self.create_subscription(Image, WRIST_CAMERA_TOPIC, self._wrist_image_cb, 10)
        self.create_subscription(JointState, JOINT_STATE_TOPIC, self._joint_state_cb, 10)

        self._action_pub = self.create_publisher(JointState, ACTION_TOPIC, 10)

        self.get_logger().info(f"Connecting to openpi server at {host}:{port} ...")
        self._policy = _websocket_client_policy.WebsocketClientPolicy(host=host, port=port)
        self.get_logger().info(f"Connected. Metadata: {self._policy.get_server_metadata()}")

        self.create_timer(1.0 / INFERENCE_HZ, self._infer)

    # ------------------------------------------------------------------
    # Subscription callbacks
    # ------------------------------------------------------------------

    def _left_image_cb(self, msg: Image) -> None:
        img = self._bridge.imgmsg_to_cv2(msg, desired_encoding="rgb8")
        with self._lock:
            self._left_image = cv2.resize(img, (224, 224))

    def _right_image_cb(self, msg: Image) -> None:
        img = self._bridge.imgmsg_to_cv2(msg, desired_encoding="rgb8")
        with self._lock:
            self._right_image = cv2.resize(img, (224, 224))

    def _wrist_image_cb(self, msg: Image) -> None:
        img = self._bridge.imgmsg_to_cv2(msg, desired_encoding="rgb8")
        with self._lock:
            self._wrist_image = cv2.resize(img, (224, 224))

    def _joint_state_cb(self, msg: JointState) -> None:
        name_to_pos = dict(zip(msg.name, msg.position))
        positions = [name_to_pos.get(n, 0.0) for n in UR5E_JOINT_NAMES]
        with self._lock:
            self._joint_positions = np.array(positions, dtype=np.float32)  # (8,)

    # ------------------------------------------------------------------
    # Inference loop
    # ------------------------------------------------------------------

    def _infer(self) -> None:
        with self._lock:
            if self._joint_positions is None:
                self.get_logger().info("Waiting for joint states ...", throttle_duration_sec=5.0)
                return

            left_img = (
                self._left_image
                if self._left_image is not None
                else np.zeros((224, 224, 3), dtype=np.uint8)
            )
            # Fall back to left image if no right camera is publishing.
            right_img = self._right_image if self._right_image is not None else left_img
            wrist_img = (
                self._wrist_image
                if self._wrist_image is not None
                else np.zeros((224, 224, 3), dtype=np.uint8)
            )

            # State: 6 joint positions + gripper (average of two finger joints) = 7-dim.
            gripper = float(self._joint_positions[6] + self._joint_positions[7]) / 2.0
            state = np.concatenate([self._joint_positions[:6], [gripper]], dtype=np.float32)

            obs = {
                "exterior_image_left": left_img,
                "exterior_image_right": right_img,
                "wrist_image": wrist_img,
                "state": state,
                "prompt": self._prompt,
            }

        try:
            result = self._policy.infer(obs)
        except Exception as exc:
            self.get_logger().error(f"Inference error: {exc}")
            return

        # actions shape: (chunk_size, 7) or (7,) — take first step.
        actions = np.asarray(result.get("actions", []))
        if actions.ndim > 1:
            actions = actions[0]

        # Model returns 7-dim: [6 joints, gripper]. Map to 8 joint commands
        # by applying the same gripper value to both finger joints.
        joint_cmd = np.concatenate([actions[:6], [actions[6], actions[6]]])

        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = UR5E_JOINT_NAMES
        msg.position = joint_cmd.tolist()
        self._action_pub.publish(msg)


def main() -> None:
    parser = argparse.ArgumentParser(description="UR5e fine-tuned pi05_ur5 ROS2 bridge")
    parser.add_argument("--host", default="localhost", help="openpi server IP or hostname")
    parser.add_argument("--port", type=int, default=8000, help="openpi server port")
    parser.add_argument("--prompt", default="pick up the object", help="Language instruction")
    parser.add_argument(
        "--right-camera-topic",
        default=RIGHT_CAMERA_TOPIC,
        help="ROS2 topic for the right exterior camera (defaults to /right_camera/color/image_raw; "
             "falls back to left image if no messages arrive)",
    )
    args, ros_args = parser.parse_known_args()

    rclpy.init(args=ros_args)
    node = UR5eFinetunedBridge(
        host=args.host,
        port=args.port,
        prompt=args.prompt,
        right_camera_topic=args.right_camera_topic,
    )
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
