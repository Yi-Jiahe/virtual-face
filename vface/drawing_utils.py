from cv2 import cv2
import numpy as np
import mediapipe as mp

from .mesh_points import MeshPoints


class MediaPipeDrawer:
    def __init__(self):
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.drawing_spec = self.mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

    def debug_out_of_the_box(self, image, results_face=None, results_hands=None, results_holistic=None):
        if results_face:
            # Draw the face mesh annotations on the image.
            if results_face.multi_face_landmarks:
                for face_landmarks in results_face.multi_face_landmarks:
                    mp.solutions.drawing_utils.draw_landmarks(
                        image=image,
                        landmark_list=face_landmarks,
                        connections=mp.solutions.face_mesh.FACEMESH_TESSELATION,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=self.mp_drawing_styles
                            .get_default_face_mesh_tesselation_style())

        if results_hands:
            if results_hands.multi_hand_landmarks:
                for hand_landmarks in results_hands.multi_hand_landmarks:
                    mp.solutions.drawing_utils.draw_landmarks(
                        image=image,
                        landmark_list=hand_landmarks,
                        connections=mp.solutions.hands.HAND_CONNECTIONS,
                        landmark_drawing_spec=self.mp_drawing_styles.get_default_hand_landmarks_style(),
                        connection_drawing_spec=self.mp_drawing_styles.get_default_hand_connections_style())

        if results_holistic:
            mp.solutions.drawing_utils.draw_landmarks(
                image=image,
                landmark_list=results_holistic.face_landmarks,
                connections=mp.solutions.holistic.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=self.mp_drawing_styles
                    .get_default_face_mesh_tesselation_style())
            # 2. Right hand
            mp.solutions.drawing_utils.draw_landmarks(
                image=image,
                landmark_list=results_holistic.right_hand_landmarks,
                connections=mp.solutions.holistic.HAND_CONNECTIONS,
                landmark_drawing_spec=self.mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=4),
                connection_drawing_spec=self.mp_drawing_styles.get_default_hand_connections_style())

            # 3. Left Hand
            mp.solutions.drawing_utils.draw_landmarks(
                image=image,
                landmark_list=results_holistic.left_hand_landmarks,
                connections=mp.solutions.holistic.HAND_CONNECTIONS,
                landmark_drawing_spec=self.mp_drawing_styles.get_default_hand_landmarks_style(),
                connection_drawing_spec=self.mp_drawing_styles.get_default_hand_connections_style())

            # 4. Pose Detections
            mp.solutions.drawing_utils.draw_landmarks(
                image=image,
                landmark_list=results_holistic.pose_landmarks,
                connections=mp.solutions.holistic.POSE_CONNECTIONS,
                landmark_drawing_spec=mp.solutions.drawing_utils.DrawingSpec(color=(245, 117, 66), thickness=2,
                                                                             circle_radius=2),
                connection_drawing_spec=mp.solutions.drawing_utils.DrawingSpec(color=(245, 66, 230), thickness=2,
                                                                               circle_radius=2))

        # Display the image
        cv2.imshow('MediaPipe FaceMesh and Hands', image)


def draw_face(image, face_landmarks):
    blank_image = np.multiply(np.ones(image.shape), (0, 0, 0))
    draw_landmarks(image, face_landmarks)

    # Display the image
    cv2.imshow('Silhouettes and Iris', image)


def draw_landmarks(image, face_landmarks):
    image_rows, image_cols, _ = image.shape
    for i, landmark in enumerate(face_landmarks.landmark):
        x_px = int(landmark.x * image_cols)
        y_px = int(landmark.y * image_rows)
        image_point = (x_px, y_px)

        draw_eye(image, i, image_point)
        draw_irises(image, i, image_point)
        draw_mouth(image, i, image_point)
        draw_silhouette(image, i, image_point)


def draw_eye(image, i, image_point):
    color = None
    if i in [*MeshPoints.leftEyeUpper0,
             *MeshPoints.leftEyeLower0,
             *MeshPoints.rightEyeLower0,
             *MeshPoints.rightEyeUpper0]:
        color = (0, 0, 255)
    if i in [*MeshPoints.leftEyeUpper1,
             *MeshPoints.leftEyeLower1,
             *MeshPoints.rightEyeLower1,
             *MeshPoints.rightEyeUpper1]:
        color = (63, 63, 255)
    if i in [*MeshPoints.leftEyeUpper2,
             *MeshPoints.leftEyeLower2,
             *MeshPoints.rightEyeLower2,
             *MeshPoints.rightEyeUpper2]:
        color = (127, 127, 255)
    if i in [*MeshPoints.leftEyeLower3,
             *MeshPoints.rightEyeLower3]:
        color = (191, 191, 255)
    if color is not None:
        cv2.circle(image, image_point, radius=2, color=color, thickness=-1)


def draw_irises(image, i, image_point):
    color = None
    if i in [*MeshPoints.leftEyeIris,
             *MeshPoints.rightEyeIris]:
        color = (0, 255, 0)
    if color is not None:
        cv2.circle(image, image_point, radius=2, color=color, thickness=-1)


def draw_mouth(image, i, image_point):
    color = None
    if i in [*MeshPoints.lipsLowerInner,
             *MeshPoints.lipsUpperInner]:
        color = (0, 0, 255)
    if i in [*MeshPoints.lipsLowerOuter,
             *MeshPoints.lipsUpperOuter]:
        color = (127, 127, 255)
    if color is not None:
        cv2.circle(image, image_point, radius=2, color=color, thickness=-1)


def draw_silhouette(image, i, image_point):
    color = None
    text = None
    if i in [*MeshPoints.silhouette]:
        color = (0, 0, 255)
        if i == 10:
            text = "top"
        if i == 152:
            text = "bottom"
        if i == 234:
            text = "left"
        if i == 454:
            text = "right"
    if color is not None:
        cv2.circle(image, image_point, radius=2, color=color, thickness=-1)
        if text is not None:
            cv2.putText(image, text, image_point, fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.3, color=color,
                        thickness=1,
                        lineType=cv2.LINE_AA)
