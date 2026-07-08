# UR5 Example

## Running the Policy Server (Remote GPU Machine)

This section describes how to serve a trained UR5 checkpoint (e.g. `pi05_ur5_cube`, fine-tuned via
`src/openpi/training/config.py`) on a remote GPU machine, and connect it to a ROS2 + Isaac Sim setup
on a local machine. Two ways to run the server are covered: directly with `uv` (simplest, no Docker
needed if the machine already has your `openpi` checkout and env set up), or via Docker (better
isolation, useful for machines you don't want to install project dependencies onto directly).

### Architecture

```
[Local machine]                                    [Remote GPU machine]
Isaac Sim  -->  ros2_isaac_sim_bridge_finetuned.py  -->  openpi policy server (uv or Docker)
               (ROS2 topics)                             (WebSocket :8000)
```

### Prerequisites

**Remote machine:**
- A trained checkpoint under `checkpoints/<config_name>/<exp_name>/<step>/` (see the main
  [README](../../README.md) for `compute_norm_stats.py` / `scripts/train.py` usage).
- For the `uv` route: `uv` installed and the repo synced (`uv sync`).
- For the Docker route: Docker installed (rootless mode recommended — see
  [docs/docker.md](../../docs/docker.md)), NVIDIA Container Toolkit, Docker Compose V2, and BuildKit
  (buildx) — see the commands below.

**Local machine:**
- ROS2 Humble sourced in your shell
- `openpi-client` available — either `uv run` from inside this repo, or standalone:
  ```bash
  pip install packages/openpi-client/
  pip install typing_extensions
  ```
- Isaac Sim running and publishing ROS2 topics

### Step 1 — Start the policy server on the remote machine

Both options serve the exact same checkpoint; pick whichever fits your remote environment.

#### Option A — via `uv` (native, no Docker)

```bash
cd openpi
uv run scripts/serve_policy.py \
    --port=8000 \
    policy:checkpoint \
    --policy.config=pi05_ur5_cube \
    --policy.dir=checkpoints/pi05_ur5_cube/cube_run/2999
```

- `--policy.config` must match a `TrainConfig` name in `src/openpi/training/config.py`.
- `--policy.dir` is the checkpoint directory to load (relative to the repo root, or absolute).
- Note the top-level flags (`--port`) come **before** the `policy:checkpoint` subcommand — passing
  them after will fail with `Unrecognized options`.
- To keep it running after closing the SSH session:
  ```bash
  nohup uv run scripts/serve_policy.py --port=8000 policy:checkpoint \
      --policy.config=pi05_ur5_cube \
      --policy.dir=checkpoints/pi05_ur5_cube/cube_run/2999 \
      > serve_policy.log 2>&1 &
  disown
  ```
- Wait for: `INFO:websockets.server:server listening on 0.0.0.0:8000`
- To stop: `pkill -f serve_policy.py`

#### Option B — via Docker

```bash
cd openpi
export SERVER_ARGS="--port=8000 policy:checkpoint --policy.config=pi05_ur5_cube --policy.dir=checkpoints/pi05_ur5_cube/cube_run/2999"
docker compose -f scripts/docker/compose.yml up --build
```

`compose.yml` bind-mounts the repo root into `/app` in the container, so a `--policy.dir` path
relative to the repo root (as above) resolves correctly without extra volume config. Model weights
for the base checkpoint (~11 GB) are downloaded from GCS on first run and cached in `~/.cache/openpi`
via the `OPENPI_DATA_HOME` volume; subsequent starts reuse the cache (omit `--build` once the image
exists).

To run in the background, press `d` while the compose output is shown, or add `-d`:
```bash
docker compose -f scripts/docker/compose.yml up -d
```

To stop:
```bash
docker stop <container_id>   # get id from: docker ps
```

One-time setup if `docker compose`/buildx aren't already available:
```bash
mkdir -p ~/.docker/cli-plugins
curl -SL https://github.com/docker/compose/releases/download/v2.24.5/docker-compose-linux-x86_64 \
    -o ~/.docker/cli-plugins/docker-compose
chmod +x ~/.docker/cli-plugins/docker-compose

curl -SL https://github.com/docker/buildx/releases/download/v0.12.1/buildx-v0.12.1.linux-amd64 \
    -o ~/.docker/cli-plugins/docker-buildx
chmod +x ~/.docker/cli-plugins/docker-buildx
```

### Step 2 — Open an SSH tunnel from the local machine (optional)

If port 8000 is not directly reachable (common on cluster nodes), tunnel it over SSH:

```bash
ssh -L 8000:localhost:8000 <user>@<remote_ip>
```

Keep this terminal open while using the bridge, and use `--host localhost` in Step 3. If the remote's
port 8000 is directly reachable from the local machine (verify with `nc -zv <remote_ip> 8000`), skip
the tunnel and use `--host <remote_ip>` directly.

### Step 3 — Run the ROS2 bridge on the local machine

Make sure Isaac Sim is publishing to the expected ROS2 topics, then run the bridge matching the
observation format the checkpoint was trained on (fine-tuned pi05_ur5 models use
`ros2_isaac_sim_bridge_finetuned.py`):

```bash
cd openpi
uv run examples/ur5/ros2_isaac_sim_bridge_finetuned.py \
    --host <remote_ip_or_localhost> \
    --port 8000 \
    --prompt "pick up the yellow cube from the table"
```

(`python3 examples/ur5/ros2_isaac_sim_bridge_finetuned.py ...` works the same if you installed
`openpi-client` standalone instead of using `uv run`.)

The bridge subscribes to:
| Topic | Type | Description |
|---|---|---|
| `/rgb_left` | `sensor_msgs/Image` | Left exterior camera |
| `/rgb_right` | `sensor_msgs/Image` | Right exterior camera (falls back to left if not published) |
| `/rgb_wrist` | `sensor_msgs/Image` | Wrist camera |
| `/joint_states` | `sensor_msgs/JointState` | Current joint positions |

And publishes to:
| Topic | Type | Description |
|---|---|---|
| `joint_command` | `sensor_msgs/JointState` | Commanded joint positions (absolute; the server converts the model's predicted deltas back to absolute using the state you publish — see `AbsoluteActions` in `src/openpi/transforms.py`) |

Joint order: `shoulder_pan_joint`, `shoulder_lift_joint`, `elbow_joint`, `wrist_1_joint`, `wrist_2_joint`, `wrist_3_joint`, plus the two gripper finger joints.

Useful flags:
- `--exec-horizon N` (default `16`) — how many actions from each predicted chunk to execute before re-querying the policy for a new one. Lower = more reactive/closed-loop, more server calls; higher = fewer calls, more open-loop.
- `--debug` — logs each new action chunk's shape and first/last rows.
- `--save-images` — dumps the first inference's camera frames to `debug_images/` for sanity-checking what the policy actually sees.

---

## Fine-tuning on UR5 Data

Below we provide an outline of how to implement the key components mentioned in the "Finetune on your data" section of the [README](../README.md) for finetuning on UR5 datasets.

First, we will define the `UR5Inputs` and `UR5Outputs` classes, which map the UR5 environment to the model and vice versa. Check the corresponding files in `src/openpi/policies/libero_policy.py` for comments explaining each line.

```python

@dataclasses.dataclass(frozen=True)
class UR5Inputs(transforms.DataTransformFn):

    model_type: _model.ModelType = _model.ModelType.PI0

    def __call__(self, data: dict) -> dict:
        # First, concatenate the joints and gripper into the state vector.
        state = np.concatenate([data["joints"], data["gripper"]])

        # Possibly need to parse images to uint8 (H,W,C) since LeRobot automatically
        # stores as float32 (C,H,W), gets skipped for policy inference.
        base_image = _parse_image(data["base_rgb"])
        wrist_image = _parse_image(data["wrist_rgb"])

        # Create inputs dict.
        inputs = {
            "state": state,
            "image": {
                "base_0_rgb": base_image,
                "left_wrist_0_rgb": wrist_image,
                # Since there is no right wrist, replace with zeros
                "right_wrist_0_rgb": np.zeros_like(base_image),
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                # Since the "slot" for the right wrist is not used, this mask is set
                # to False
                "right_wrist_0_rgb": np.True_ if self.model_type == _model.ModelType.PI0_FAST else np.False_,
            },
        }

        if "actions" in data:
            inputs["actions"] = data["actions"]

        # Pass the prompt (aka language instruction) to the model.
        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class UR5Outputs(transforms.DataTransformFn):

    def __call__(self, data: dict) -> dict:
        # Since the robot has 7 action dimensions (6 DoF + gripper), return the first 7 dims
        return {"actions": np.asarray(data["actions"][:, :7])}

```

Next, we will define the `UR5DataConfig` class, which defines how to process raw UR5 data from LeRobot dataset for training. For a full example, see the `LeRobotLiberoDataConfig` config in the [training config file](https://github.com/physical-intelligence/openpi/blob/main/src/openpi/training/config.py).

```python

@dataclasses.dataclass(frozen=True)
class LeRobotUR5DataConfig(DataConfigFactory):

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        # Boilerplate for remapping keys from the LeRobot dataset. We assume no renaming needed here.
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "base_rgb": "image",
                        "wrist_rgb": "wrist_image",
                        "joints": "joints",
                        "gripper": "gripper",
                        "prompt": "prompt",
                    }
                )
            ]
        )

        # These transforms are the ones we wrote earlier.
        data_transforms = _transforms.Group(
            inputs=[UR5Inputs(action_dim=model_config.action_dim, model_type=model_config.model_type)],
            outputs=[UR5Outputs()],
        )

        # Convert absolute actions to delta actions.
        # By convention, we do not convert the gripper action (7th dimension).
        delta_action_mask = _transforms.make_bool_mask(6, -1)
        data_transforms = data_transforms.push(
            inputs=[_transforms.DeltaActions(delta_action_mask)],
            outputs=[_transforms.AbsoluteActions(delta_action_mask)],
        )

        # Model transforms include things like tokenizing the prompt and action targets
        # You do not need to change anything here for your own dataset.
        model_transforms = ModelTransformFactory()(model_config)

        # We return all data transforms for training and inference. No need to change anything here.
        return dataclasses.replace(
            self.create_base_config(assets_dirs),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
        )

```

Finally, we define the TrainConfig for our UR5 dataset. Here, we define a config for fine-tuning pi0 on our UR5 dataset. See the [training config file](https://github.com/physical-intelligence/openpi/blob/main/src/openpi/training/config.py) for more examples, e.g. for pi0-FAST or for LoRA fine-tuning.

```python
TrainConfig(
    name="pi0_ur5",
    model=pi0.Pi0Config(),
    data=LeRobotUR5DataConfig(
        repo_id="your_username/ur5_dataset",
        # This config lets us reload the UR5 normalization stats from the base model checkpoint.
        # Reloading normalization stats can help transfer pre-trained models to new environments.
        # See the [norm_stats.md](../docs/norm_stats.md) file for more details.
        assets=AssetsConfig(
            assets_dir="gs://openpi-assets/checkpoints/pi0_base/assets",
            asset_id="ur5e",
        ),
        base_config=DataConfig(
            # This flag determines whether we load the prompt (i.e. the task instruction) from the
            # ``task`` field in the LeRobot dataset. The recommended setting is True.
            prompt_from_task=True,
        ),
    ),
    # Load the pi0 base model checkpoint.
    weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_base/params"),
    num_train_steps=30_000,
)
```





