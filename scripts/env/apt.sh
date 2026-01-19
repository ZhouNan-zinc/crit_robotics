#!/usr/bin/env bash

echo "[env] Installing apt dependencies (requires sudo)"

sudo apt update
sudo apt install -y \
  build-essential \
  wget \
  cmake \
  python3-colcon-common-extensions \
  python3-rosdep \
  python3-vcstool \
  libboost-all-dev \
  protobuf-compiler \
  libprotobuf-dev
