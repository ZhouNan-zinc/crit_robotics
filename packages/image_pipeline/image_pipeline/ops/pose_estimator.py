import cv2



BIG_ARMOR_POSE = []

SMALL_ARMOR_POSE = []

BASE_POSE = []

def pose_estimate(keypoints, class_id):
    if class_id % 10 == 1: # HERO
        pass
