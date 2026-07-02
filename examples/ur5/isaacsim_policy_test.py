"""Isaac Sim policy test — no ROS2.

Paste into the Isaac Sim Script Editor and run.
Reads joint states and camera images directly via Isaac Sim APIs,
calls the openpi policy server, and applies actions via articulation control.

Adjust the CONFIG section to match your scene prim paths.
"""

import sys
sys.path.insert(0, "/home/graal/isaacsim/kit/python/lib/python3.11/site-packages")

import numpy as np

# ---------------------------------------------------------------------------
# CONFIG — adjust to your scene
# ---------------------------------------------------------------------------
ROBOT_PRIM = "/mir"
LEFT_CAM   = "/mir/base_link_cabinet/cabinet/realsense_d455_left"
RIGHT_CAM  = "/mir/base_link_cabinet/cabinet/realsense_d455_right"
WRIST_CAM  = "/mir/onrobot_2fg7/camera_holder_d405/camera_holder/realsense_d405_wrist"
HOST       = "localhost"
PORT       = 8000
PROMPT     = "pick up the red cube from the table"
# ---------------------------------------------------------------------------

from omni.isaac.core import World
from omni.isaac.core.articulations import ArticulationView
from omni.isaac.sensor import Camera
from openpi_client import websocket_client_policy as _websocket_client_policy

policy = _websocket_client_policy.WebsocketClientPolicy(host=HOST, port=PORT)
print(f"Connected to policy server at {HOST}:{PORT}")

world = World.instance()

robot = ArticulationView(prim_paths_expr=ROBOT_PRIM)
robot.initialize()

def _make_cam(path):
    cam = Camera(prim_path=path)
    cam.initialize()
    return cam

left_cam  = _make_cam(LEFT_CAM)
right_cam = _make_cam(RIGHT_CAM)
wrist_cam = _make_cam(WRIST_CAM)

chunk       = None
tick        = 0
gripper_bin = 1.0  # 1.0=open, matching episode-start convention


def _get_rgb(cam):
    try:
        rgba = cam.get_rgba()
        if rgba is not None:
            return rgba[:, :, :3]
    except Exception:
        pass
    return np.zeros((224, 224, 3), dtype=np.uint8)


def on_physics_step(dt):
    global chunk, tick, gripper_bin

    joints = robot.get_joint_positions()[0]  # (8,)
    state  = np.concatenate([joints[:6], [gripper_bin]], dtype=np.float32)

    if chunk is None or tick // 2 >= len(chunk):
        obs = {
            "exterior_image_left":  _get_rgb(left_cam),
            "exterior_image_right": _get_rgb(right_cam),
            "wrist_image":          _get_rgb(wrist_cam),
            "state":                state,
            "prompt":               PROMPT,
        }
        try:
            result = policy.infer(obs)
        except Exception as exc:
            print(f"Inference error: {exc}")
            return

        actions = np.asarray(result.get("actions", []), dtype=np.float32)
        if actions.ndim == 1:
            actions = actions[np.newaxis, :]
        print(f"New chunk: shape={actions.shape}  actions[0]={actions[0]}  actions[-1]={actions[-1]}")
        chunk = actions
        tick  = 0

    action_7    = chunk[tick // 2]
    tick       += 1

    gripper_bin = 1.0 if float(action_7[6]) >= 0.5 else 0.0
    # Isaac Sim convention: 0.0=open, 1.0=closed → invert training convention
    gripper_cmd = 1.0 - gripper_bin

    cmd = np.concatenate([action_7[:6], [gripper_cmd, gripper_cmd]])
    robot.set_joint_position_targets(cmd[np.newaxis, :])


world.add_physics_callback("openpi_step", on_physics_step)
print("Policy callback registered.")
