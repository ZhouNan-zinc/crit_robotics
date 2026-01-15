# Getting Started

This guide walks through setting up the Crit Robotics workspace on a fresh machine. If you prefer a zero-config workflow, open the repository in VS Code and select **Reopen in Container**; the `.devcontainer/devcontainer.json` image (`osrf/ros:kilted-desktop`) already contains ROS 2 and GPU tooling.

## 1. Prerequisites

- Ubuntu 22.04 (recommended) with NVIDIA drivers if you plan to run GPU inference.
- ROS 2 Humble Hawksbill (or Rolling) installed under `/opt/ros/${ROS_DISTRO}`.
- Git, curl, and build essentials (`build-essential`, `cmake`, etc.).
- (Camera) Hikrobot MVS SDK installed and available on your library path if you intend to run `unicam`.
- (Python) [uv](https://github.com/astral-sh/uv) for deterministic dependency management.

## 2. Clone

```bash
mkdir -p ~/workspaces && cd ~/workspaces
git clone https://github.com/<org>/crit_robotics.git
cd crit_robotics
```

## 3. Shell Setup

Run once to append ROS and workspace overlays to your shell startup file:

```bash
./scripts/setup.bash
```

This script adds the following snippet to `~/.bashrc`:

```bash
source /opt/ros/${ROS_DISTRO}/setup.bash
if [ -f "install/setup.bash" ]; then
  . "install/setup.bash"
fi
```

## 4. Python Dependencies

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.local/share/uv/venv/bin/activate   # or add to PATH per installer output
uv pip sync
```

`uv` reads `pyproject.toml`/`uv.lock` and installs OpenVINO, OpenCV, lap, numba, etc. into an isolated virtual environment.

## 5. Build the Workspace

```bash
colcon build --symlink-install
source install/setup.bash
```

If you see missing dependencies, install them via `apt` (e.g., `ros-${ROS_DISTRO}-image-transport`). Re-run `colcon build` afterwards.

## 6. Launch Components

### Camera

```bash
ros2 launch unicam launch.py camera:=hikcam config:=default.yaml
```

Parameters such as `pixel_width`, `exposure_time`, and `gain` live in `packages/unicam/config/default.yaml`. Calibration YAML files sit under `packages/unicam/calibration/`.

### Perception Pipeline

```bash
ros2 run imagepipe imagepipe --ros-args -p model_name_or_path:=/path/to/ov_model
```

Optional parameters:

- `subscribe_to`: list of camera namespaces (default `['hikcam']`).
- `device`: OpenVINO device string (`CPU`, `GPU`, `AUTO`).
- `dtype`: `float32` or `bf16`.
- `conf_thres`/`iou_thres`: detection thresholds.

### UDP Bridge

```bash
ros2 run udp_socket udp_socket --ros-args --params-file packages/udp_socket/config/default.yaml
```

Publish to `test_send` or listen on `test_recv` to verify connectivity.

## 7. Testing & Linting

```bash
colcon test
colcon test --packages-select imagepipe  # Python-only package
```

Add unit or integration tests whenever you introduce new behavior. The C++ packages enable `-Wall -Wextra -Wpedantic`; fix warnings introduced by your changes.

## 8. Troubleshooting

| Symptom | Fix |
| --- | --- |
| `FindMVS.cmake` cannot locate Hikrobot SDK | Export `MVS_SDK_ROOT` or edit `packages/unicam/cmake/FindMVS.cmake` with the correct install prefix. |
| OpenVINO complains about missing `config.json` | Ensure the model directory contains `config.json` describing `input_size`. See `OpenVinoBackend` constructor for required fields. |
| `image_transport_py` import error | Install `ros-${ROS_DISTRO}-image-transport-plugins` and `python3-image-transport` via apt. |
| `UDP socket bind: Permission denied` | Make sure you are not binding to privileged ports (<1024) and no other process owns the chosen port. |

Have another trick that saved you time? Please add it to this document!
