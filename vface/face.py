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
        "iris_ratio": {
            "x": 0,
            "y": 0
        },
        "mouth_aspect_ratio": 0
    }

    top, bottom, left, right = None, None, None, None
    left_eye_landmarks = [None for _ in range(5)]
    right_eye_landmarks = [None for _ in range(5)]
    left_iris = None
    right_iris = None
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
        # Left Iris
        if i == 468:
            left_iris = landmark
        if i == 473:
            right_iris = landmark
        # Mouth
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
    face_data["pose"]["roll"], face_data["pose"]["pitch"], face_data["pose"]["yaw"] \
        = determine_face_pose(top, bottom, left, right)
    face_data["position"]["x"] = round(np.mean((left.x, right.x)), 4)
    face_data["position"]["y"] = round(np.mean((left.y, right.y)), 4)
    face_data["position"]["z"] = round(np.mean((left.z, right.z)), 4)
    face_data["eye_aspect_ratio"]["left"], face_data["eye_aspect_ratio"]["right"], \
        face_data["iris_ratio"]["x"], face_data["iris_ratio"]["y"] \
        = eye_parameters(left_eye_landmarks, right_eye_landmarks, left_iris, right_iris)
    face_data["mouth_aspect_ratio"] = mouth_aspect_ratio(mouth_landmarks)

    return face_data


def determine_face_pose(top, bottom, left, right):
    roll = round(np.rad2deg(np.arctan2(left.y - right.y, right.x - left.x)), 4)
    pitch = round(np.rad2deg(np.arctan2(top.z - bottom.z, bottom.y - top.y)), 4)
    yaw = round(np.rad2deg(np.arctan2(left.z - right.z, right.x - left.x)), 4)
    return roll, pitch, yaw


def eye_parameters(left_eye_landmarks, right_eye_landmarks, left_iris, right_iris):
    eye_aspect_ratios = [None, None]
    iris_ratios = [[None, None], [None, None]]
    for i, (eye_landmarks, iris) in enumerate(zip((left_eye_landmarks, right_eye_landmarks), (left_iris, right_iris))):
        # Since there are no points in the MediaPipe model that nicely line up when the eye is closed,
        # we will use the average of two points on the top to meet the bottom
        outer, top_outer, top_inner, inner, bottom, iris \
            = map(lambda p: np.array([p.x, p.y, p.z]), (*eye_landmarks, iris))
        top = (top_outer + top_inner) / 2

        eye_aspect_ratios[i] = eye_aspect_ratio(outer, inner, top, bottom)
        iris_ratios[i] = iris_ratio(outer, inner, top, bottom, iris)
        # Take the average of the iris positions, correcting the x value to be 0 => left, 1 => right
    return *eye_aspect_ratios, (iris_ratios[0][0] + (1 - iris_ratios[1][0])) / 2, (iris_ratios[0][1] + iris_ratios[1][1]) / 2


def eye_aspect_ratio(outer, inner, top, bottom):
    # Ratio of the height to the length of the eye
    # Measure of how open the eye is
    SCALING_FACTOR = 1.5
    return np.round(np.linalg.norm(top - bottom) / np.linalg.norm(outer - inner) * SCALING_FACTOR, 4)


def iris_ratio(outer, inner, top, bottom, iris):
    # Relative position of the iris in the eye
    # x: 0 => outer, 1 => inner
    # y: 0 => bottom, 1 => top
    ratio_x = np.dot(iris-outer, inner-outer) / np.linalg.norm(inner-outer)**2
    ratio_y = np.dot(iris-bottom, top-bottom) / np.linalg.norm(top-bottom)**2
    return ratio_x, ratio_y


def mouth_aspect_ratio(mouth_landmarks):
    p1, p2, p3, p4, p5, p6, p7, p8 = map(lambda p: np.array([p.x, p.y, p.z]), mouth_landmarks)
    return np.round(
        (np.linalg.norm(p2 - p8) + np.linalg.norm(p3 - p7) + np.linalg.norm(p4 - p6)) / (2 * np.linalg.norm(p1 - p5)),
        4)
