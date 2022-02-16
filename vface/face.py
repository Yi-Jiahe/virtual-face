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
        },
        "eye_aspect_ratio": {
            "left": 0,
            "right": 0
        },
        "mouth_aspect_ratio": 0
    }

    top, bottom, left, right = None, None, None, None
    left_eye_landmarks = [None for _ in range(5)]
    right_eye_landmarks = [None for _ in range(5)]
    mouth_landmarks = [None for _ in range(8)]
    for i, landmark in enumerate(face_landmarks.landmark):
        # Silhouette landmarks
        if i == 10:
            top = landmark
        if i == 152:
            bottom = landmark
        if i == 234:
            left = landmark
        if i == 454:
            right = landmark
        # Left eye
        if i == 33:
            left_eye_landmarks[0] = landmark
        if i == 159:
            left_eye_landmarks[1] = landmark
        if i == 158:
            left_eye_landmarks[2] = landmark
        if i == 133:
            left_eye_landmarks[3] = landmark
        if i == 145:
            left_eye_landmarks[4] = landmark
        # Right eye
        if i == 263:
            right_eye_landmarks[0] = landmark
        if i == 386:
            right_eye_landmarks[1] = landmark
        if i == 385:
            right_eye_landmarks[2] = landmark
        if i == 362:
            right_eye_landmarks[3] = landmark
        if i == 374:
            right_eye_landmarks[4] = landmark
        if i == 78:
            mouth_landmarks[0] = landmark
        if i == 80:
            mouth_landmarks[1] = landmark
        if i == 13:
            mouth_landmarks[2] = landmark
        if i == 311:
            mouth_landmarks[3] = landmark
        if i == 308:
            mouth_landmarks[4] = landmark
        if i == 402:
            mouth_landmarks[5] = landmark
        if i == 14:
            mouth_landmarks[6] = landmark
        if i == 88:
            mouth_landmarks[7] = landmark
    face_data["pose"]["roll"], face_data["pose"]["pitch"], face_data["pose"]["yaw"] = determine_face_pose(top, bottom, left, right)
    face_data["position"]["x"] = round(np.mean((left.x, right.x)), 4)
    face_data["position"]["y"] = round(np.mean((left.y, right.y)), 4)
    face_data["position"]["z"] = round(np.mean((left.z, right.z)), 4)
    face_data["eye_aspect_ratio"]["left"], face_data["eye_aspect_ratio"]["right"] = eye_aspect_ratio(left_eye_landmarks, right_eye_landmarks)
    face_data["mouth_aspect_ratio"] = mouth_aspect_ratio(mouth_landmarks)

    return face_data


def determine_face_pose(top, bottom, left, right):
    roll = round(np.rad2deg(np.arctan2(right.y - left.y, right.x - left.x)), 4)
    pitch = round(np.rad2deg(np.arctan2(top.z - bottom.z, bottom.y - top.y)), 4)
    yaw = round(np.rad2deg(np.arctan2(right.z - left.z, right.x - left.x)), 4)
    return roll, pitch, yaw


def eye_aspect_ratio(left_eye_landmarks, right_eye_landmarks):
    SCALING_FACTOR = 1.5
    eye_aspect_ratios = [None, None]
    for i, landmarks in enumerate((left_eye_landmarks, right_eye_landmarks)):
        outer, top_outer, top_inner, inner, bottom = map(lambda p: np.array([p.x, p.y, p.z]), landmarks)
        eye_aspect_ratios[i] = np.linalg.norm((top_outer+top_inner)/2 - bottom) / np.linalg.norm(outer-inner) * SCALING_FACTOR
    return eye_aspect_ratios


def mouth_aspect_ratio(mouth_landmarks):
    p1, p2, p3, p4, p5, p6, p7, p8 = map(lambda p: np.array([p.x, p.y, p.z]), mouth_landmarks)
    return (np.linalg.norm(p2-p8) + np.linalg.norm(p3-p7) + np.linalg.norm(p4-p6)) / (2 * np.linalg.norm(p1-p5))