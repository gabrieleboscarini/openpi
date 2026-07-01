#!/usr/bin/env python3
"""ROS2 bridge for the fine-tuned pi05_ur5 policy.

Matches the observation format used during training on gabbobosca/mir_ur_pick_place:
  - exterior_image_left  (base/left camera)
  - exterior_image_right (right exterior camera; falls back to left if not published)
  - wrist_image
  - state (7-dim: 6 joint positions + binary gripper)

Single 60 Hz timer. ActionChunkBroker handles chunk replay and re-queries the
policy every ACTION_HORIZON steps. Images are center-cropped to 224x224 before
being sent to the server.

Run:
    python3 examples/ur5/ros2_isaac_sim_bridge_finetuned.py \
        --host localhost --port 8000 \
        --prompt "pick up the red cube from the table"
"""

import argparse

import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import Image, JointState

from openpi_client import websocket_client_policy as _websocket_client_policy
from openpi_client.action_chunk_broker import ActionChunkBroker
from openpi_client.image_tools import resize_with_pad

np.set_printoptions(precision=4, suppress=True)

LEFT_CAMERA_TOPIC = "/rgb_left"
RIGHT_CAMERA_TOPIC = "/rgb_right"
WRIST_CAMERA_TOPIC = "/rgb_wrist"
JOINT_STATE_TOPIC  = "/joint_states"
ACTION_TOPIC       = "joint_command"

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

EXECUTION_HZ   = 60  # physics rate
ACTION_HORIZON = 10  # steps between policy re-queries (10/60 Hz ≈ 167 ms)


class UR5eFinetunedBridge(Node):
    def __init__(self, host: str, port: int, prompt: str,
                 left_camera_topic: str, right_camera_topic: str, wrist_camera_topic: str,
                 debug: bool = False) -> None:
        super().__init__("ur5e_finetuned_bridge")

        self._bridge = CvBridge()
        self._prompt = prompt
        self._debug  = debug

        self._left_image:     np.ndarray | None = None
        self._right_image:    np.ndarray | None = None
        self._wrist_image:    np.ndarray | None = None
        self._joint_positions: np.ndarray | None = None

        # Binary gripper state fed to the model (1.0=open, 0.0=closed).
        # Initialized open, matching the episode-start convention in training data.
        self._gripper_state: float = 1.0

        self.create_subscription(Image, left_camera_topic,  self._left_image_cb,  10)
        self.create_subscription(Image, right_camera_topic, self._right_image_cb, 10)
        self.create_subscription(Image, wrist_camera_topic, self._wrist_image_cb, 10)
        self.create_subscription(JointState, JOINT_STATE_TOPIC, self._joint_state_cb, 10)

        self._action_pub = self.create_publisher(JointState, ACTION_TOPIC, 10)

        self.get_logger().info(f"Connecting to openpi server at {host}:{port} ...")
        ws_policy = _websocket_client_policy.WebsocketClientPolicy(host=host, port=port)
        self._policy = ActionChunkBroker(ws_policy, action_horizon=ACTION_HORIZON)
        self.get_logger().info("Connected.")

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
        self._joint_positions = np.array(
            [name_to_pos.get(n, 0.0) for n in UR5E_JOINT_NAMES], dtype=np.float32
        )

    # ------------------------------------------------------------------
    # Main step (60 Hz)
    # ------------------------------------------------------------------

    def _step(self) -> None:
        if self._joint_positions is None or self._left_image is None or self._wrist_image is None:
            self.get_logger().info("Waiting for sensors...", throttle_duration_sec=5.0)
            return

        left_img  = self._left_image
        right_img = self._right_image if self._right_image is not None else left_img
        wrist_img = self._wrist_image
        state = np.concatenate([self._joint_positions[:6], [self._gripper_state]], dtype=np.float32)

        obs = {
            "exterior_image_left":  left_img,
            "exterior_image_right": right_img,
            "wrist_image":          wrist_img,
            "state":                state,
            "prompt":               self._prompt,
        }

        try:
            result = self._policy.infer(obs)
        except Exception as exc:
            self.get_logger().error(f"Inference error: {exc}")
            return

        action_7 = np.asarray(result["actions"], dtype=np.float32)

        if self._debug:
            self.get_logger().info(f"action={action_7}")

        self._gripper_state = 1.0 if float(action_7[6]) >= 0.5 else 0.0

        # Training: 1.0=open, 0.0=closed → Isaac Sim: 0.0=open, 1.0=closed → invert.
        gripper_cmd = 1.0 - self._gripper_state
        joint_cmd = np.concatenate([action_7[:6], [gripper_cmd, gripper_cmd]])

        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = UR5E_JOINT_NAMES
        msg.position = joint_cmd.tolist()
        self._action_pub.publish(msg)


def main() -> None:
    parser = argparse.ArgumentParser(description="UR5e fine-tuned pi05_ur5 ROS2 bridge")
    parser.add_argument("--host",   default="localhost")
    parser.add_argument("--port",   type=int, default=8000)
    parser.add_argument("--prompt", default="pick up the red cube from the table")
    parser.add_argument("--left-camera-topic",  default=LEFT_CAMERA_TOPIC)
    parser.add_argument("--right-camera-topic", default=RIGHT_CAMERA_TOPIC)
    parser.add_argument("--wrist-camera-topic", default=WRIST_CAMERA_TOPIC)
    parser.add_argument("--debug",  action="store_true")
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
    )
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
