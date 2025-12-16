#!/usr/bin/env bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_DIR="$SCRIPT_DIR/env"

print_step() {
  echo -e "\n[setup] $1" >&2
}

print_step "Robot Operating System"
"$ENV_DIR/install_ros.sh"

print_step "ROS environment"
"$ENV_DIR/rosdep.sh"

print_step "System dependencies"
"$ENV_DIR/apt.sh"

print_step "Shell configuration"
"$ENV_DIR/bashrc.sh"

print_step "Python dependencies"
"$ENV_DIR/pyenv.sh"

print_step "Hardware dependencies"
"$ENV_DIR/install_mvs.sh"

print_step "Done."
