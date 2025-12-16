"""Pose estimation helpers for projective geometry routines."""

import cv2
import numpy as np
from typing import Optional, Sequence, Tuple

BIG_ARMOR_POINTS: Sequence[Tuple[float, float, float]] = []
SMALL_ARMOR_POINTS: Sequence[Tuple[float, float, float]] = []
BASE_POINTS: Sequence[Tuple[float, float, float]] = []

CLASS_TO_POINTS = {
    1: BIG_ARMOR_POINTS,
    6: BIG_ARMOR_POINTS,
    2: SMALL_ARMOR_POINTS,
    4: SMALL_ARMOR_POINTS,
    5: BASE_POINTS,
}

def pose_estimate(keypoints: np.ndarray,
                  class_id: int,
                  camera_matrix: np.ndarray,
                  distortion_coefficients: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Estimate the 6DoF pose of an object given image points and class ID.

    Returns position and orientation.
    """
    object_points = np.asarray(CLASS_TO_POINTS.get(class_id, SMALL_ARMOR_POINTS), dtype=np.float32)
    keypoints = np.asarray(keypoints, dtype=np.float32).reshape(-1, 2)
    try:
        is_success, rvec, tvec = cv2.solvePnP(
            objectPoints=object_points,
            imagePoints=keypoints,
            cameraMatrix=camera_matrix,
            distCoeffs=distortion_coefficients,
            flags=cv2.SOLVEPNP_IPPE
        )

        if not is_success:
            raise RuntimeError("Solve PNP failed")

    except RuntimeError:
        return None, None

    rvec = rvec.reshape(3)
    tvec = tvec.reshape(3)

    x = tvec[0]
    y = tvec[1]
    z = tvec[2]

    R, _ = cv2.Rodrigues(rvec)
    sy = np.sqrt(R[0, 0]**2 + R[1, 0]**2) # roll, pitch, yaw (around X, Y, Z)
    singular = sy < 1e-6

    if not singular:
        r  = np.arctan2(R[2, 1], R[2, 2])
        p = np.arctan2(-R[2, 0], sy)
        y   = np.arctan2(R[1, 0], R[0, 0])
    else:
        # Gimbal lock
        r  = np.arctan2(-R[1, 2], R[1, 1])
        p = np.arctan2(-R[2, 0], sy)
        y   = 0.0

    position = np.array([x, y, z])
    orientation = np.array([r, p, y])
    return position, orientation
