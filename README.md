# Crit Robotics Workspace

Crit Robotics is a ROS 2 workspace that bundles the camera drivers, perception nodes, and networking utilities that power our competitive robotics experiments.  It is built for ROS 2 Humble (or newer) with OpenVINO-backed perception pipelines and includes a ready-to-run VS Code dev container.

## Highlights

- **Camera bring-up** &mdash; `unicam` wraps Hikrobot USB cameras, publishes `image_raw` and calibrated `sensor_msgs/CameraInfo`, and exposes runtime tuning parameters.
- **Perception pipeline** &mdash; `imagepipe` performs OpenVINO end-to-end YOLO inference, ByteTrack multi-object tracking, and pose estimation that feeds `vision_msgs/Detection2DArray`.
- **Networking bridge** &mdash; `udp_socket` provides UDP broadcast helpers plus protobuf scaffolding so detections/commands can hop networks.
- **Custom interfaces** &mdash; `control_msgs` defines the minimal command schema used by downstream gimbal or turret controllers.

## Getting Started (TL;DR)

```bash
# Clone and open in VS Code or your preferred editor
# Optional but recommended: use the built-in dev container definition

# If you are *not* using the dev container, install ROS 2 Humble and run:
./scripts/setup.bash   # appends ROS sourcing instructions to ~/.bashrc
sudo apt update && sudo apt install -y python3-vcstool python3-colcon-common-extensions
curl -LsSf https://astral.sh/uv/install.sh | sh && source ~/.local/share/uv/venv/bin/activate
uv pip sync

colcon build --symlink-install
source install/setup.bash

# Example launch files
ros2 launch unicam launch.py camera=hikcam
ros2 run imagepipe imagepipe     # consumes the /hikcam/image_raw topic
ros2 run udp_socket udp_socket --ros-args --params-file packages/udp_socket/config/default.yaml
```

See `docs/getting-started.md` for the full walkthrough, including GPU requirements, calibration details, and troubleshooting tips.

## Repository Layout

| Path | Description |
| --- | --- |
| `crit_robotics/` | Aggregate meta-package, launchers, and URDF placeholders. |
| `packages/unicam/` | Hikrobot USB camera driver (C++). Includes calibration YAML and launch/config helpers. |
| `packages/imagepipe/` | Python perception stack: OpenVINO backend, YOLO detection, ByteTrack tracking, pose estimation. |
| `packages/udp_socket/` | UDP networking node plus protobuf schema for telemetry bridging. |
| `packages/control_msgs/` | Custom ROS 2 interfaces (messages, future services). |
| `scripts/` | Environment bootstrap (`setup.bash`) and utility tools (ballistic table generator). |
| `docs/` | Additional guides (architecture, onboarding). |
| `.devcontainer/` | VS Code dev container targeting `osrf/ros:kilted-desktop` with GPU pass-through. |

## Development Workflow

1. **Environment** &mdash; Use the dev container for a reproducible GPU-enabled setup, or follow `docs/getting-started.md` to install ROS 2 + system deps manually.
2. **Python dependencies** &mdash; Use [uv](https://github.com/astral-sh/uv) to sync `pyproject.toml`/`uv.lock` so OpenVINO, NumPy, etc. stay pinned.
3. **C++ dependencies** &mdash; Install vendor SDKs (e.g., Hikrobot MVS) and verify `packages/unicam/cmake/FindMVS.cmake` can resolve them.
4. **Build** &mdash; `colcon build --symlink-install` from the workspace root. Rebuild affected packages after code changes.
5. **Test** &mdash; Run `colcon test` (C++) and `pytest` (Python). Add tests alongside new features when possible.
6. **Lint** &mdash; `ament_lint_auto` and `ament_flake8`/`ament_pep257` run during CI. Match the pedantic compiler flags defined in each package.

## Launch Recipes

- `ros2 launch unicam launch.py camera:=hikcam` &mdash; start the Hikrobot camera and publish `/hikcam/image_raw`.
- `ros2 run imagepipe imagepipe` &mdash; run inference on whichever namespaces are listed in the `subscribe_to` parameter (defaults to `hikcam`).
- `ros2 run udp_socket udp_socket` &mdash; bring up the UDP bridge and echo packets across the network.
- `ros2 launch crit_robotics bringup.launch.py` &mdash; placeholder for combined bring-up (currently empty; extend as the system grows).

## Documentation & Support

- `docs/getting-started.md` &mdash; environment setup, building, and running nodes.
- `docs/architecture.md` &mdash; deeper dive into how the packages interact.
- `CONTRIBUTING.md` &mdash; coding standards, review expectations, release checklist.
- `CODE_OF_CONDUCT.md` &mdash; community expectations.

Questions, bug reports, or feature ideas? Open an issue using the provided templates once the repo is published, or reach out directly to the maintainers listed in `package.xml`.
