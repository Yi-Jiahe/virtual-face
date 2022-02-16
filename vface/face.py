import numpy as np


def extract_pose_data(face_landmarks):
    face_data = {
        "pose": {
            "roll": 0,
            "pitch": 0,
            "yaw": 0
        },
        "position": {
            "x": 0,
            "y": 0,
            "z": 0
        }
    }

    top, bottom, left, right = None, None, None, None
    for i, landmark in enumerate(face_landmarks.landmark):
        if i == 10:
            top = landmark
        if i == 152:
            bottom = landmark
        if i == 234:
            left = landmark
        if i == 454:
            right = landmark
    face_data["pose"]["roll"] = round(np.rad2deg(np.arctan2(right.y - left.y, right.x - left.x)), 4)
    face_data["pose"]["pitch"] = round(-np.rad2deg(np.arctan2(bottom.z - top.z, bottom.y - top.y)), 4)
    face_data["pose"]["yaw"] = round(np.rad2deg(np.arctan2(right.z - left.z, right.x - left.x)), 4)
    face_data["position"]["x"] = round(np.mean((left.x, right.x)), 4)
    face_data["position"]["y"] = round(np.mean((left.y, right.y)), 4)
    face_data["position"]["z"] = round(np.mean((left.z, right.z)), 4)

    return face_data