#!/usr/bin/env bash

BASHRC="$HOME/.bashrc"
MARKER_BEGIN="# >>> ros2 workspace setup >>>"

echo "[env] Configuring ~/.bashrc"

if ! grep -qF "$MARKER_BEGIN" "$BASHRC"; then
  if [ -s "$BASHRC" ] && [ -n "$(tail -n 1 "$BASHRC")" ]; then
    echo >> "$BASHRC"
  fi

  cat <<'EOF' >> "$BASHRC"
# >>> ros2 workspace setup >>>
source /opt/ros/${ROS_DISTRO}/setup.bash

if [ -f "install/setup.bash" ]; then
  # shellcheck disable=SC1091
  . "install/setup.bash"
fi
# <<< ros2 workspace setup <<<
EOF
fi
