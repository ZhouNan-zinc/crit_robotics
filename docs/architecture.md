# Architecture Overview

Crit Robotics is organized as a ROS 2 workspace with a set of focused packages. This document explains how the pieces fit together so that new contributors can orient themselves quickly.

```
┌────────────┐    sensor_msgs/Image     ┌─────────────────────────┐
│  unicam    │ ───────────────────────▶ │   image_pipeline node   │
│  (C++)     │    camera_info + params  │ • OpenVINO YOLO         │
│            │                          │ • ByteTrack             │
└────────────┘                          │ • Pose estimator        │
                                        └────────────┬────────────┘
                                                     │ vision_msgs/Detection2DArray
                                                     ▼
                                           ┌──────────────────┐
                                           │  Consumers (e.g. │
                                           │  targeting UI,   │
                                           │  control nodes)  │
                                           └──────────────────┘
```

## Packages

### crit_robotics (meta-package)

- Provides workspace-level launch files (currently stubs) and serves as the aggregation point for future robot description assets (URDF/meshes).
- Depends on all downstream packages for easier bring-up.

### unicam

- C++ ROS 2 package responsible for interfacing with Hikrobot USB cameras via the vendor MVS SDK.
- Publishes `image_transport` camera topics, handles calibration through `camera_info_manager`, and exposes tuning parameters such as exposure, gain, gamma, and ROI sizing.
- Launch: `ros2 launch unicam launch.py camera:=hikcam config:=default.yaml`.
- Key files:
  - `include/unicam/node_interface.hpp` – common base class for camera nodes.
  - `src/hik_usb_cam.cpp` – Hikvision-specific implementation.
  - `config/default.yaml` – parameter presets per namespace.

### image_pipeline

- Python ROS 2 node that consumes images and camera info topics.
- `image_pipeline/pipe/ov_end2end_yolo.py` loads OpenVINO compiled models, performs inference, applies `end2end_fastnms`, and feeds detections into a ByteTrack multi-object tracker.
- `image_pipeline/node/pose_estimate.py` converts tracker output into `vision_msgs/Detection2DArray` messages, estimating 6-DoF pose via `cv2.solvePnP`.
- Depends on GPU/CPU OpenVINO runtime (configurable via ROS parameters), `lap` (assignment), and `numba`.
- Designed to be extended with additional pipelines that inherit from `PosePipelineNodeInterface`.

### udp_socket

- Boost.Asio-based ROS 2 node for UDP broadcast/receive.
- Exposes a `register_callback` hook so application-specific logic can be layered on top; `ProtobufLayer` shows how to convert between UDP payloads and `std_msgs/String`.
- `proto/message.proto` holds the canonical Pose schema. Extend it as telemetry/data needs grow.

### control_msgs

- Custom ROS interface definitions shared by motion controllers.
- Currently contains `Command.msg` (pitch, yaw, frequency, enable flag); future services/actions can live here.

## Data Flow

1. `unicam` publishes `/<camera_ns>/image_raw` and calibrated `CameraInfo` using the namespaces defined in `config/default.yaml`.
2. `image_pipeline` subscribes to one or more namespaces (parameter `subscribe_to`), performs detection + tracking, and publishes `detection_array` messages.
3. Consumers (gimbal control, UI overlays, telemetry exporters) use the detection array, optionally bridging it over the network via `udp_socket`.
4. Commands to actuators are expressed using `control_msgs/Command` for consistency.

## Launching Multiple Packages

In the future, `crit_robotics/launch/bringup.launch.py` will coordinate the camera, perception, networking, and control stack. Until then, launch each package individually or craft a temporary launch file tailored to your hardware layout.

## Extending the System

- **New cameras** – Derive from `CameraNodeInterface`, implement `run()` and `dynamic_reconfigure()`, then wire up parameters in `config/` and `launch/`.
- **New perception pipelines** – Implement a new node inheriting from `PosePipelineNodeInterface` (or `ImagePipelineNodeInterface`) and expose ROS parameters for tuning.
- **Networking formats** – Modify `packages/udp_socket/proto/` and regenerate bindings (future work) or simply reinterpret the byte strings inside `register_callback`.
- **Control interfaces** – Add messages/services under `packages/control_msgs/msg|srv` and remember to update the `CMakeLists.txt` and `package.xml` dependencies.

For deeper instructions about environment setup, GPU configuration, and troubleshooting, head to `docs/getting-started.md`.
