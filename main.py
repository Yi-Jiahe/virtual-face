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


def connect_socket(sock, addr):
    connected = False
    while not connected:
        try:
            sock.connect(addr)
            print(f"Connected to {addr[0], addr[1]}")
        except ConnectionRefusedError:
            print(f"Failed to connect to {addr[0], addr[1]}")
            print("Retrying...")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run detection and server.')
    parser.add_argument('--host', dest='host',
                        default="127.0.0.1",
                        help='Server host to listen on')
    parser.add_argument('-p', '--port', dest='port', type=int,
                        default=12345,
                        help='Port to listen on')
    parser.add_argument("-d", "--debug", help="show debug window",
                        action="store_true")
    parser.add_argument("-s", "--standalone", help="run in standalone mode",
                        action="store_true")
    args = parser.parse_args()

    debug = args.debug
    standalone = args.standalone

    signal.signal(signal.SIGINT, set_killed)
    signal.signal(signal.SIGTERM, set_killed)

    if not standalone:
        address = (args.host, args.port)
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        connect_socket(sock, address)

    if debug:
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

            if not standalone:
                try:
                    msg = '%.4f %.4f %.4f %.4f %.4f %.4f' % \
                          (face_data["pose"]["roll"], face_data["pose"]["pitch"], face_data["pose"]["yaw"], 0, 0, 0)
                    sock.send(bytes(msg, "utf-8"))
                except OSError:
                    # Socket is not connected
                    # Attempt to reconnect
                    print("Failed to send data")
                    print("Attempting to reconnect")
                    connect_socket(sock, address)

            if debug:
                # Prepare image for drawing on and displaying
                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                drawn_image = image.copy()
                drawer.debug_out_of_the_box(drawn_image, results_holistic=results)

                if results.face_landmarks:
                    face_landmarks = results.face_landmarks
                    draw_face(drawn_image, face_landmarks)

                cv2.imshow("debug", drawn_image)

                print(json.dumps(face_data))

                # Terminate the process
                if cv2.waitKey(5) & 0xFF == 27:
                    break

    if not standalone:
        sock.close()

    cap.release()
    if debug:
        cv2.destroyAllWindows()