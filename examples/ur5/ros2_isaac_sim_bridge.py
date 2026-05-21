#!/usr/bin/env python3
"""ROS2 bridge: connects an Isaac Sim UR5e to the openpi WebSocket policy server.

Subscribes to Isaac Sim camera and joint-state topics, sends observations to the
openpi server, and publishes the returned joint commands back to the simulator.

Run alongside your Isaac Sim scene:
    python ros2_isaac_sim_bridge.py --host <server_ip> --port 8000 \
        --prompt "pick up the red cube"

Dependencies (install in your ROS2 workspace):
    pip install openpi-client opencv-python-headless numpy
    # openpi-client is at packages/openpi-client in this repo
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
BASE_CAMERA_TOPIC = "/camera/color/image_raw"
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


class UR5eOpenPIBridge(Node):
    def __init__(self, host: str, port: int, prompt: str) -> None:
        super().__init__("ur5e_openpi_bridge")

        self._bridge = CvBridge()
        self._prompt = prompt
        self._lock = threading.Lock()

        self._base_image: np.ndarray | None = None
        self._wrist_image: np.ndarray | None = None
        self._joint_positions: np.ndarray | None = None

        self.create_subscription(Image, BASE_CAMERA_TOPIC, self._base_image_cb, 10)
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

    def _base_image_cb(self, msg: Image) -> None:
        img = self._bridge.imgmsg_to_cv2(msg, desired_encoding="rgb8")
        with self._lock:
            self._base_image = cv2.resize(img, (224, 224))

    def _wrist_image_cb(self, msg: Image) -> None:
        img = self._bridge.imgmsg_to_cv2(msg, desired_encoding="rgb8")
        with self._lock:
            self._wrist_image = cv2.resize(img, (224, 224))

    def _joint_state_cb(self, msg: JointState) -> None:
        name_to_pos = dict(zip(msg.name, msg.position))
        positions = [name_to_pos.get(n, 0.0) for n in UR5E_JOINT_NAMES]
        with self._lock:
            self._joint_positions = np.array(positions, dtype=np.float64)  # (8,)

    # ------------------------------------------------------------------
    # Inference loop
    # ------------------------------------------------------------------

    def _infer(self) -> None:
        with self._lock:
            if self._joint_positions is None:
                self.get_logger().info(
                    "Waiting for joint states ...", throttle_duration_sec=5.0
                )
                return

            base_img = (
                self._base_image
                if self._base_image is not None
                else np.zeros((224, 224, 3), dtype=np.uint8)
            )
            wrist_img = (
                self._wrist_image
                if self._wrist_image is not None
                else np.zeros((224, 224, 3), dtype=np.uint8)
            )

            obs = {
                "observation/exterior_image_1_left": base_img,
                "observation/wrist_image_left": wrist_img,
                "observation/joint_position": self._joint_positions,
                "observation/gripper_position": np.zeros(1, dtype=np.float64),
                "prompt": self._prompt,
            }

        try:
            result = self._policy.infer(obs)
        except Exception as exc:
            self.get_logger().error(f"Inference error: {exc}")
            return

        # actions shape is either (action_dim,) or (chunk_size, action_dim).
        actions = np.asarray(result.get("actions", []))
        if actions.ndim > 1:
            actions = actions[0]

        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = UR5E_JOINT_NAMES
        msg.position = actions[:8].tolist()
        self._action_pub.publish(msg)


def main() -> None:
    parser = argparse.ArgumentParser(description="UR5e ↔ openpi ROS2 bridge")
    parser.add_argument("--host", default="localhost", help="openpi server IP or hostname")
    parser.add_argument("--port", type=int, default=8000, help="openpi server port")
    parser.add_argument("--prompt", default="pick up the object", help="Language instruction")
    args, ros_args = parser.parse_known_args()

    rclpy.init(args=ros_args)
    node = UR5eOpenPIBridge(host=args.host, port=args.port, prompt=args.prompt)
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
