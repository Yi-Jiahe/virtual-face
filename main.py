import socket
import threading
import json
import os

from cv2 import cv2
import mediapipe as mp
from dotenv import load_dotenv

from vface.drawing_utils import draw_face, MediaPipeDrawer
from vface.face import extract_pose_data


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


def accept_connections(s, exit_event):
    while True:
        try:
            conn, addr = s.accept()  # Establish connection with client.
            print(f"Got connection from {addr[0]}:{addr[1]}")
            threading.Thread(target=send_data, args=(conn, addr)).start()
        except BlockingIOError:
            pass
        if exit_event.is_set():
            break


def send_data(conn, addr):
    with conn:
        while True:
            try:
                data = conn.recv(1024)
                if data:
                    conn.send(json.dumps(face_data).encode('utf-8'))
                    print(f"Sent to {addr[0]}:{addr[1]}")
                else:
                    print(f"Connection closed by {addr[0]}:{addr[1]}")
                    break
            except BlockingIOError:
                pass


if __name__ == '__main__':
    load_dotenv()

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    host = "127.0.0.1"
    port = int(os.getenv("SOCKET_PORT"))
    sock.bind((host, port))
    sock.listen()
    sock.setblocking(False)

    exit_event = threading.Event()
    socket_thread = threading.Thread(target=accept_connections, args=(sock, exit_event))
    socket_thread.start()

    drawer = MediaPipeDrawer()

    cap = cv2.VideoCapture(0)

    # DETECT THE FACE LANDMARKS
    with mp.solutions.face_mesh.FaceMesh(refine_landmarks=True, min_detection_confidence=0.8,
                               min_tracking_confidence=0.8) as face_mesh, \
            mp.solutions.hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.8) as hands, \
            mp.solutions.holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while True:
            success, image = cap.read()

            # Flip the image horizontally and convert the color space from BGR to RGB
            image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)

            # To improve performance, mark image as not writeable before processing to pass by reference
            image.flags.writeable = False

            # Detect the landmarks
            results_face = face_mesh.process(image)
            results_hands = hands.process(image)
            results_holistic = holistic.process(image)

            # Prepare image for drawing on and displaying
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            # drawer.debug_out_of_the_box(image, results_face, results_hands)
            drawer.debug_out_of_the_box(image.copy(), results_holistic=results_holistic)

            if results_face.multi_face_landmarks:
                face_landmarks = results_face.multi_face_landmarks[0]
                draw_face(image.copy(), face_landmarks)
                face_data = extract_pose_data(face_landmarks)

            # Terminate the process
            if cv2.waitKey(5) & 0xFF == 27:
                break

    exit_event.set()
    sock.close()

    cap.release()
    cv2.destroyAllWindows()