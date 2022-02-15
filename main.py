import argparse
import signal
import socket
import threading
import json

from cv2 import cv2
import mediapipe as mp

from vface.drawing_utils import draw_face, MediaPipeDrawer
from vface.face import extract_pose_data


killed = False


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


def set_killed(signum, frame):
    print("Killing program")
    global killed
    killed = True


def accept_connections(s, exit_event):
    while not killed:
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
        while not killed:
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
    parser = argparse.ArgumentParser(description='Run detection and server.')
    parser.add_argument('-p', '--port', dest='port', type=int,
                        default=12345,
                        help='Port to listen on')
    parser.add_argument("-d", "--debug", help="show debug window",
                        action="store_true")
    args = parser.parse_args()

    signal.signal(signal.SIGINT, set_killed)
    signal.signal(signal.SIGTERM, set_killed)

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    host = "127.0.0.1"
    port = args.port
    sock.bind((host, port))
    print(f"listening at {host}:{port}")
    sock.listen()
    sock.setblocking(False)

    exit_event = threading.Event()
    socket_thread = threading.Thread(target=accept_connections, args=(sock, exit_event))
    socket_thread.start()

    drawer = MediaPipeDrawer()

    cap = cv2.VideoCapture(0)

    # DETECT THE FACE LANDMARKS
    with mp.solutions.holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while not killed:
            success, image = cap.read()

            # Flip the image horizontally and convert the color space from BGR to RGB
            image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)

            # To improve performance, mark image as not writeable before processing to pass by reference
            image.flags.writeable = False

            # Detect the landmarks
            results = holistic.process(image)

            if results.face_landmarks:
                face_landmarks = results.face_landmarks
                face_data = extract_pose_data(face_landmarks)

            if args.debug:
                # Prepare image for drawing on and displaying
                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                drawn_image = image.copy()
                drawer.debug_out_of_the_box(drawn_image, results_holistic=results)

                if results.face_landmarks:
                    face_landmarks = results.face_landmarks
                    draw_face(drawn_image, face_landmarks)

                cv2.imshow("debug", drawn_image)

                # Terminate the process
                if cv2.waitKey(5) & 0xFF == 27:
                    break

    exit_event.set()
    sock.close()

    cap.release()
    cv2.destroyAllWindows()