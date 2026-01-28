"""Pose estimation helpers for projective geometry routines."""

import cv2
import numpy as np
from typing import Optional, Sequence, Tuple

# HACK: This is not a simplified version. For RoboMaster only.
def pose_estimate(keypoints: np.ndarray,
                  object_points: np.ndarray,
                  camera_matrix: np.ndarray,
                  distortion_coefficients: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Estimate the 6DoF pose of an object given image points and class ID.

    Returns position and orientation.
    """
    import logging
    logging.warning(f"{keypoints}")
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

    except:
        return None, None
    
    rvec = rvec.reshape(3)
    tvec = tvec.reshape(3)

    x = tvec[0]
    y = tvec[1]
    z = tvec[2]

    R, _ = cv2.Rodrigues(rvec)

    R_x = np.array([
        [1, 0, 0,],
        [0, np.cos(np.deg2rad(15)), -np.sin(np.deg2rad(15))],
        [0, np.sin(np.deg2rad(15)), np.cos(np.deg2rad(15))],
    ])

    R = R @ R_x

    logging.warning(f"{R}")

    roll = np.arctan2(-R[0, 1], R[0, 0])
    pitch = np.deg2rad(-15)
    
    yaw = np.arctan2(R[0,2], abs(R[2,2])) * 1.105 # np.arctan2(sintheta, costheta)

    position = np.array([x, y, z])
    orientation = np.array([roll, pitch, yaw])

    return position, orientation
