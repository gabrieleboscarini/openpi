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
        --prompt "pick up the red cube from the table"

Two-rate control loop:
  - INFERENCE_HZ (10 Hz): queries the policy server, stores the returned action chunk.
  - EXECUTION_HZ (30 Hz): steps through the stored chunk one action at a time,
    matching the 30 fps rate at which the training data was collected.
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

np.set_printoptions(precision=4, suppress=True)

# ---------------------------------------------------------------------------
# Topic names — adjust to match your Isaac Sim ROS2 bridge configuration.
# ---------------------------------------------------------------------------
LEFT_CAMERA_TOPIC = "/rgb_left"
RIGHT_CAMERA_TOPIC = "/rgb_right"
WRIST_CAMERA_TOPIC = "/rgb_wrist"
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

# Policy server queries at 10 Hz; chunk execution replays at 30 Hz (training fps).
INFERENCE_HZ = 10
EXECUTION_HZ = 30


class UR5eFinetunedBridge(Node):
    def __init__(self, host: str, port: int, prompt: str,
                 left_camera_topic: str, right_camera_topic: str, wrist_camera_topic: str,
                 debug: bool = False) -> None:
        super().__init__("ur5e_finetuned_bridge")

        self._bridge = CvBridge()
        self._prompt = prompt
        self._debug = debug
        self._lock = threading.Lock()
        self._infer_count = 0

        self._left_image: np.ndarray | None = None
        self._right_image: np.ndarray | None = None
        self._wrist_image: np.ndarray | None = None
        # 8-dim: [shoulder_pan, shoulder_lift, elbow, wrist_1, wrist_2, wrist_3, left_finger, right_finger]
        self._joint_positions: np.ndarray | None = None
        # Binary gripper state fed to the model: 1.0=open, 0.0=closed.
        # Training default is 1.0 (open) at episode start; updated after each
        # inference by thresholding the model's gripper output.
        self._gripper_state: float = 1.0

        # Action chunk returned by the policy: shape (chunk_size, 7).
        # Stepped through at EXECUTION_HZ; replaced at INFERENCE_HZ.
        self._action_chunk: np.ndarray | None = None
        self._chunk_index: int = 0

        self.create_subscription(Image, left_camera_topic, self._left_image_cb, 10)
        self.create_subscription(Image, right_camera_topic, self._right_image_cb, 10)
        self.create_subscription(Image, wrist_camera_topic, self._wrist_image_cb, 10)
        self.create_subscription(JointState, JOINT_STATE_TOPIC, self._joint_state_cb, 10)

        self._action_pub = self.create_publisher(JointState, ACTION_TOPIC, 10)

        self.get_logger().info(f"Connecting to openpi server at {host}:{port} ...")
        self._policy = _websocket_client_policy.WebsocketClientPolicy(host=host, port=port)
        self.get_logger().info(f"Connected. Metadata: {self._policy.get_server_metadata()}")

        # Execution timer runs at 30 Hz, inference timer at 10 Hz.
        self.create_timer(1.0 / EXECUTION_HZ, self._publish_action)
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
    # Execution loop (30 Hz) — publish one action from the stored chunk
    # ------------------------------------------------------------------

    def _publish_action(self) -> None:
        with self._lock:
            if self._action_chunk is None:
                return
            if self._chunk_index >= len(self._action_chunk):
                # Chunk exhausted; hold the last action until the next inference.
                action_7 = self._action_chunk[-1]
            else:
                action_7 = self._action_chunk[self._chunk_index]
                self._chunk_index += 1

        # Model returns 7-dim: [6 joints, gripper]. Map to 8 joint commands
        # by applying the same gripper value to both finger joints.
        joint_cmd = np.concatenate([action_7[:6], [action_7[6], action_7[6]]])

        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = UR5E_JOINT_NAMES
        msg.position = joint_cmd.tolist()
        self._action_pub.publish(msg)

    # ------------------------------------------------------------------
    # Inference loop (10 Hz) — query server and refresh the action chunk
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

            # State: 6 joint positions + gripper (binary 1.0=open / 0.0=closed) = 7-dim.
            # Training uses event-based binary gripper, not raw finger joint radians.
            state = np.concatenate([self._joint_positions[:6], [self._gripper_state]], dtype=np.float32)

            obs = {
                "exterior_image_left": left_img,
                "exterior_image_right": right_img,
                "wrist_image": wrist_img,
                "state": state,
                "prompt": self._prompt,
            }

        self._infer_count += 1
        n = self._infer_count

        has_left  = obs["exterior_image_left"].any()
        has_right = obs["exterior_image_right"].any()
        has_wrist = obs["wrist_image"].any()
        self.get_logger().info(
            f"[infer #{n}] state={obs['state']}  "
            f"cameras: left={'OK' if has_left else 'ZERO'} "
            f"right={'OK' if has_right else 'ZERO'} "
            f"wrist={'OK' if has_wrist else 'ZERO'}"
        )

        # Policy inference happens outside the lock so image/joint callbacks
        # can continue updating while we wait for the server response.
        try:
            result = self._policy.infer(obs)
        except Exception as exc:
            self.get_logger().error(f"Inference error: {exc}")
            return

        # actions shape: (chunk_size, 7)
        actions = np.asarray(result.get("actions", []), dtype=np.float32)
        if actions.ndim == 1:
            actions = actions[np.newaxis, :]  # (1, 7)

        self.get_logger().info(
            f"[infer #{n}] chunk shape={actions.shape}  "
            f"actions[1]={actions[1] if len(actions) > 1 else actions[0]}  "
            f"actions[5]={actions[5] if len(actions) > 5 else 'n/a'}  "
            f"actions[-1]={actions[-1]}"
        )
        if self._debug:
            self.get_logger().info(f"[infer #{n}] full chunk:\n{actions}")

        # Update binary gripper state from model output (threshold at 0.5).
        # Use the middle of the chunk as a stable estimate.
        mid = len(actions) // 2
        self._gripper_state = 1.0 if float(actions[mid, 6]) >= 0.5 else 0.0
        self.get_logger().info(f"[infer #{n}] gripper_state={self._gripper_state}  (raw mid={actions[mid, 6]:.3f})")

        with self._lock:
            self._action_chunk = actions
            # Start from index 1: delta[0] == 0 by training convention
            # (action[t] == state[t] in the dataset), so skip it.
            self._chunk_index = 1


def main() -> None:
    parser = argparse.ArgumentParser(description="UR5e fine-tuned pi05_ur5 ROS2 bridge")
    parser.add_argument("--host", default="localhost", help="openpi server IP or hostname")
    parser.add_argument("--port", type=int, default=8000, help="openpi server port")
    parser.add_argument(
        "--prompt",
        default="pick up the red cube from the table",
        help="Language instruction (match training distribution, e.g. 'pick up the red cube from the table')",
    )
    parser.add_argument("--left-camera-topic", default=LEFT_CAMERA_TOPIC,
                        help="ROS2 topic for the left/base camera")
    parser.add_argument("--right-camera-topic", default=RIGHT_CAMERA_TOPIC,
                        help="ROS2 topic for the right exterior camera (falls back to left if no messages arrive)")
    parser.add_argument("--wrist-camera-topic", default=WRIST_CAMERA_TOPIC,
                        help="ROS2 topic for the wrist camera")
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print full action chunk on every inference call",
    )
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
