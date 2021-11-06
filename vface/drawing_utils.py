from cv2 import cv2

from .mesh_points import MeshPoints


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
