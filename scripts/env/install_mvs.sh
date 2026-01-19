#!/usr/bin/env bash

if [[ -n "${MVCAM_SDK_PATH:-}" ]]; then
  echo "[env] MVS SDK has been installed."
  exit 0
fi

DEB_URL="https://github.com/serianrealm/crit_robotics/releases/download/mvs-runtime-4.6.1-linux-x86_64/MVS-4.6.1_x86_64_20251113.deb"
DEB_PATH="/tmp/$(basename "$DEB_URL")"

echo "[env] Downloading $(basename "$DEB_URL") to /tmp ..."
wget -O "$DEB_PATH" "$DEB_URL"

echo "[env] Installing package via dpkg ..."
sudo dpkg -i "$DEB_PATH"

echo "[env] Cleaning up downloaded package ..."
rm -f "$DEB_PATH"
