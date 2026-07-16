"""Isaac Sim policy test — standalone (no ROS2, no Script Editor).

Run from VSCode / a terminal with Isaac Sim's bundled Python, e.g.:
    /home/graal/isaacsim/python.sh examples/ur5/isaacsim_policy_test_standalone.py \
        --host 130.251.6.23 --prompt "pick up the yellow cube from the table"

Unlike isaacsim_policy_test.py (which assumes a scene is already open in a
running Isaac Sim GUI session and is pasted into the Script Editor), this
script boots Isaac Sim itself, opens USD_PATH, and drives the simulation loop
directly — so it must import isaacsim.SimulationApp before any other omni.*
module.

Reads joint states and camera images directly via Isaac Sim APIs, calls the
openpi policy server, and applies actions via articulation control. Same
control logic as isaacsim_policy_test.py: observe once, block on inference,
then execute the full returned chunk (each action held for 2 physics steps)
before observing again.

Adjust the CONFIG section to match your scene.
"""

import argparse
import sys

_parser = argparse.ArgumentParser(description="Standalone Isaac Sim UR5e policy test")
_parser.add_argument("--host", required=True, help="Policy server host (e.g. localhost or a remote IP)")
_parser.add_argument("--port", type=int, default=8000, help="Policy server port")
_parser.add_argument("--prompt", default="pick up the yellow cube from the table",
                      help="Language instruction sent to the policy on every inference call")
_args, _unknown_args = _parser.parse_known_args()
# SimulationApp() re-scans sys.argv on its own and forwards whatever's left to
# the native Kit process, which doesn't know about --prompt and crashes on it
# if we don't strip our own args out first.
sys.argv = [sys.argv[0], *_unknown_args]

from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": False})  # noqa: E402 — must run before other omni imports

import numpy as np  # noqa: E402

# ---------------------------------------------------------------------------
# CONFIG — adjust to your scene
# ---------------------------------------------------------------------------
USD_PATH  = "/home/graal/isaacsim_ws/mir_ur_station/simulation/usd/robots/robot_ros_bridge.usd"
# Articulation root for the UR5e arm + 2FG7 gripper (NOT /mir — that's the
# mobile base's separate articulation; verified via stage inspection).
ROBOT_PRIM = "/mir/base_link_cabinet/cabinet/ur_mount/ur5e_physics"
LEFT_CAM   = "/mir/joints/base_link_to_realsense_right_joint/realsense_d455_left"
RIGHT_CAM  = "/mir/base_link_cabinet/cabinet/realsense_d455_right"
WRIST_CAM  = "/mir/onrobot_2fg7/camera_holder_d405/camera_holder/realsense_d405_wrist"
HOST       = _args.host
PORT       = _args.port
PROMPT     = _args.prompt
PHYSICS_DT = 1.0 / 60.0  # matches the project's Isaac Sim scene and ROS2 bridge
# ---------------------------------------------------------------------------

from omni.isaac.core import World  # noqa: E402
from omni.isaac.core.articulations import ArticulationView  # noqa: E402
from omni.isaac.core.utils.stage import open_stage  # noqa: E402
from omni.isaac.sensor import Camera  # noqa: E402
from openpi_client import websocket_client_policy as _websocket_client_policy  # noqa: E402
from openpi_client.image_tools import resize_with_pad  # noqa: E402

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

open_stage(USD_PATH)

world = World(physics_dt=PHYSICS_DT, rendering_dt=PHYSICS_DT, stage_units_in_meters=1.0)
world.reset()

robot = ArticulationView(prim_paths_expr=ROBOT_PRIM)
robot.initialize()

# The view's internal DOF order isn't guaranteed to match UR5E_JOINT_NAMES, so
# look it up by name instead of assuming a fixed slice/order.
dof_names = list(robot.dof_names)
try:
    joint_idx = [dof_names.index(name) for name in UR5E_JOINT_NAMES]
except ValueError as exc:
    raise RuntimeError(f"Joint {exc} not found in articulation dof_names={dof_names}") from exc

# The USD stage's own authored joint drives move the arm to its scene-defined
# ready pose once physics starts stepping (the same motion you'd see hitting
# Play in the GUI). For SETTLE_SECONDS we don't issue any commands at all, so
# that motion runs untouched; only after it's had time to finish does the
# policy take over and start commanding targets itself.
SETTLE_SECONDS = 3.0
SETTLE_TICKS = int(SETTLE_SECONDS / PHYSICS_DT)


def _make_cam(path: str) -> Camera:
    cam = Camera(prim_path=path)
    cam.initialize()
    return cam


left_cam = _make_cam(LEFT_CAM)
right_cam = _make_cam(RIGHT_CAM)
wrist_cam = _make_cam(WRIST_CAM)

policy = _websocket_client_policy.WebsocketClientPolicy(host=HOST, port=PORT)
print(f"Connected to policy server at {HOST}:{PORT}")

chunk        = None
tick         = 0
gripper_bin  = 0.0  # 0.0=open, matching episode-start convention
startup_tick = 0


def _get_rgb(cam: Camera) -> np.ndarray:
    try:
        rgba = cam.get_rgba()
        if rgba is not None:
            return resize_with_pad(rgba[:, :, :3], 224, 224)
    except Exception:
        pass
    return np.zeros((224, 224, 3), dtype=np.uint8)


def on_physics_step(dt: float) -> None:
    global chunk, tick, gripper_bin, startup_tick

    if startup_tick < SETTLE_TICKS:
        if startup_tick == 0:
            print(f"Letting the scene's startup motion play out for {SETTLE_SECONDS}s "
                  f"before handing control to the policy ...")
        # Issue no commands here — the USD stage's own joint drives are
        # already moving the arm toward its authored ready pose; we just let
        # physics step without interference.
        startup_tick += 1
        if startup_tick == SETTLE_TICKS:
            print("Settle complete — policy now in control.")
        return

    raw_positions = robot.get_joint_positions()[0]  # view's own DOF order
    joints = raw_positions[joint_idx][:6]  # 6 arm joints, in UR5E_JOINT_NAMES order
    state = np.concatenate([joints, [gripper_bin]], dtype=np.float32)

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
        tick = 0

    action_7 = chunk[tick // 2]
    tick += 1

    gripper_bin = 1.0 if float(action_7[6]) >= 0.5 else 0.0
    # Training and Isaac Sim share the same convention (0.0=open, 1.0=closed)
    # — no inversion needed.
    gripper_cmd = gripper_bin

    cmd_ordered = np.concatenate([action_7[:6], [gripper_cmd, gripper_cmd]])
    targets = raw_positions.copy()
    for name, value in zip(UR5E_JOINT_NAMES, cmd_ordered):
        targets[dof_names.index(name)] = value
    robot.set_joint_position_targets(targets[np.newaxis, :])


world.add_physics_callback("openpi_step", on_physics_step)
print("Policy callback registered. Running simulation loop (Ctrl+C to stop) ...")

try:
    while simulation_app.is_running():
        world.step(render=True)
except KeyboardInterrupt:
    pass
finally:
    simulation_app.close()
