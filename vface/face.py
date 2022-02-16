import numpy as np


class ParameterEstimator:
    def __init__(self):
        pass


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

    landmarks = face_landmarks.landmark

    top, bottom, left, right = \
        landmarks[10], \
        landmarks[152], \
        landmarks[234], \
        landmarks[454]
    left_eye_landmarks = [
        landmarks[33],
        landmarks[159],
        landmarks[158],
        landmarks[133],
        landmarks[145]
    ]
    right_eye_landmarks = [
        landmarks[263],
        landmarks[386],
        landmarks[385],
        landmarks[362],
        landmarks[374]
    ]
    left_iris = landmarks[468]
    right_iris = landmarks[473]
    mouth_landmarks = [
        landmarks[78],
        landmarks[80],
        landmarks[13],
        landmarks[311],
        landmarks[308],
        landmarks[402],
        landmarks[14],
        landmarks[88]
    ]

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
    """
    Returns the pose of the face based on landmarks on the silhouette of the face
    """
    roll = round(np.rad2deg(np.arctan2(left.y - right.y, right.x - left.x)), 4)
    pitch = round(np.rad2deg(np.arctan2(top.z - bottom.z, bottom.y - top.y)), 4)
    yaw = round(np.rad2deg(np.arctan2(left.z - right.z, right.x - left.x)), 4)
    return roll, pitch, yaw


def eye_parameters(left_eye_landmarks, right_eye_landmarks, left_iris, right_iris):
    """
    Takes in landmarks around the eye and the iris, returning the eye aspect ratio for each eye and iris ratio
    """
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
    """
    Takes in points around the eye and returns the eye aspect ratio (EAR)
    
    The EAR is a ratio of the height to the length of the eye and is a measure of how open the eye is
    """
    SCALING_FACTOR = 1.5
    return np.round(np.linalg.norm(top - bottom) / np.linalg.norm(outer - inner) * SCALING_FACTOR, 4)


def iris_ratio(outer, inner, top, bottom, iris):
    """
    Takes in points around the eye and the iris and returns the iris ratio

    The iris ratio is the relative position of the iris in the eye
    x: 0 => outer, 1 => inner
    y: 0 => bottom, 1 => top
    """
    ratio_x = proj(iris-outer, inner-outer) / np.linalg.norm(inner-outer)
    ratio_y = proj(iris-bottom, top-bottom) / np.linalg.norm(top-bottom)
    return ratio_x, ratio_y


def proj(v1, v2):
    """
    Takes in two vectors and returns the magnitude of the projection of v1 on v2
    """
    return np.dot(v1, v2) / np.linalg.norm(v2)


def mouth_aspect_ratio(mouth_landmarks):
    """
    Takes in an array of landmarks around the mouth and returns the mouth aspect ratio (MAR)

    The MAR is a measure of how open the mouth is
    """
    p1, p2, p3, p4, p5, p6, p7, p8 = map(lambda p: np.array([p.x, p.y, p.z]), mouth_landmarks)
    return np.round(
        (np.linalg.norm(p2 - p8) + np.linalg.norm(p3 - p7) + np.linalg.norm(p4 - p6)) / (2 * np.linalg.norm(p1 - p5)),
        4)
