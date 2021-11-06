import socket
import threading
import json
import time
import os

from cv2 import cv2
import mediapipe as mp
import numpy as np
from dotenv import load_dotenv

from vface.drawing_utils import draw_landmarks


def debug_out_of_the_box(image):
    # To improve performance
    image.flags.writeable = True

    # Convert back to the BGR color space
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # Draw the face mesh annotations on the image.
    if results_face.multi_face_landmarks:
        for face_landmarks in results_face.multi_face_landmarks:
            mp_drawing.draw_landmarks(
                image=image,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles
                    .get_default_face_mesh_tesselation_style())

    if results_hands.multi_hand_landmarks:
        for hand_landmarks in results_hands.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                image,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())

    # Display the image
    cv2.imshow('MediaPipe FaceMesh and Hands', image)


def debug(image, results_face):
    # To improve performance
    image.flags.writeable = True

    # Convert back to the BGR color space
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    blank_image = np.multiply(np.ones(image.shape), (0, 0, 0))
    if results_face.multi_face_landmarks:
        for face_landmarks in results_face.multi_face_landmarks:
            draw_landmarks(image, face_landmarks)

    # Display the image
    cv2.imshow('Silhouettes and Iris', image)


def accept_connections(s):
    threads = []
    while True:
        c, addr = s.accept()  # Establish connection with client.
        print("Got connection from", addr)
        threads.append(threading.Thread(target=send_data, args=(c,)).start())


def send_data(c):
    while True:
        c.send(json.dumps(face_data).encode('utf-8'))
        time.sleep(0.1)


if __name__ == '__main__':
    load_dotenv()

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


    s = socket.socket()
    host = socket.gethostname()
    port = int(os.getenv("SOCKET_PORT"))
    s.bind((host, port))
    s.listen()

    socket_thread = threading.Thread(target=accept_connections, args=(s,))
    socket_thread.start()

    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

    mp_face_mesh = mp.solutions.face_mesh
    mp_hands = mp.solutions.hands

    cap = cv2.VideoCapture(0)

    # DETECT THE FACE LANDMARKS
    with mp_face_mesh.FaceMesh(refine_landmarks=True, min_detection_confidence=0.8,
                               min_tracking_confidence=0.8) as face_mesh, \
            mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.8) as hands:
        while True:
            success, image = cap.read()

            # Flip the image horizontally and convert the color space from BGR to RGB
            image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)

            # To improve performance
            image.flags.writeable = False

            # Detect the landmarks
            results_face = face_mesh.process(image)
            results_hands = hands.process(image)

            if results_face.multi_face_landmarks:
                for face_landmarks in results_face.multi_face_landmarks:
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

            debug(image, results_face)

            # Terminate the process
            if cv2.waitKey(5) & 0xFF == 27:
                break

    s.close()

    cap.release()
    cv2.destroyAllWindows()
