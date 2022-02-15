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
    face_data["pose"]["roll"] = np.arctan2(left.y - right.y, left.x - right.x)
    face_data["pose"]["pitch"] = np.arctan2(top.z - bottom.z, top.y - bottom.y)
    face_data["pose"]["yaw"] = np.arctan2(left.z - right.z, left.x - right.x)
    face_data["position"]["x"] = np.mean((left.x, right.x))
    face_data["position"]["y"] = np.mean((left.y, right.y))
    face_data["position"]["z"] = np.mean((left.z, right.z))

    return face_data