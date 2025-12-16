#!/usr/bin/env bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WS_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "[env] Sourcing ROS 2 if available"
if [[ -n "${ROS_DISTRO-}" && -f "/opt/ros/${ROS_DISTRO}/setup.bash" ]]; then
  # shellcheck disable=SC1090
  source "/opt/ros/${ROS_DISTRO}/setup.bash"
else
  echo "Warning: ROS 2 not sourced in current environment." >&2
fi

echo "[env] Initializing rosdep"
command -v rosdep >/dev/null 2>&1 || {
  echo "rosdep not found; install python3-rosdep first." >&2
  exit 1
}

sudo rosdep init 2>/dev/null || true
rosdep update
rosdep install --from-paths "$WS_ROOT/src" --ignore-src -y
