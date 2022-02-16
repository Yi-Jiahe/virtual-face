import argparse
import signal
import socket
import threading
import json

from cv2 import cv2
import mediapipe as mp

from client import Client
from vface.drawing_utils import draw_face, MediaPipeDrawer
from vface.face import extract_pose_data


killed = False

def send_message(client, args):
    msg = '%.4f ' * len(args) % args
    client.send(bytes(msg, "utf-8"))


def set_killed(signum, frame):
    print("Killing program")
    global killed
    killed = True


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
        client = Client(args.host, args.port)
        client.connect()

    if debug:
        drawer = MediaPipeDrawer()

    cap = cv2.VideoCapture(0)

    with mp.solutions.holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5, refine_face_landmarks=True) as holistic:
        while not killed:
            success, image = cap.read()

            # Flip the image horizontally and convert the color space from BGR to RGB
            # image = cv2.resize(cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB), (2540, 1440))
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
                        x_ratio_left, y_ratio_left, x_ratio_right, y_ratio_right, \
                            mar, mouth_distance = 0, 0, 0, 0, 0, 0
                        send_message(client, (face_data["pose"]["roll"], face_data["pose"]["pitch"], face_data["pose"]["yaw"],
                                              face_data["eye_aspect_ratio"]["left"], face_data["eye_aspect_ratio"]["right"],
                                              x_ratio_left, y_ratio_left, x_ratio_right, y_ratio_right,
                                              mar, mouth_distance))
                    except OSError:
                        # Socket is not connected
                        # Attempt to reconnect
                        print("Failed to send data")
                        print("Attempting to reconnect")
                        client.connect()

            if debug:
                # Prepare image for drawing on and displaying
                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                image = cv2.resize(image, (1920, 1080))
                # drawer.debug_out_of_the_box(image, results_holistic=results)

                if results.face_landmarks:
                    face_landmarks = results.face_landmarks
                    draw_face(image, face_landmarks)

                cv2.imshow("debug", image)

                print(json.dumps(face_data))

                # Terminate the process
                if cv2.waitKey(5) & 0xFF == 27:
                    break

    if not standalone:
        client.close_socket()

    cap.release()
    if debug:
        cv2.destroyAllWindows()