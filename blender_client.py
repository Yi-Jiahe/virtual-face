import bpy
import socket
import threading
import json
import time
from mathutils import Matrix


def receive_from_socket(s):
    while True:
        s.send(bytes("Hit me", "utf-8"))
        face_data = json.loads(s.recv(4096))
        time.sleep(5)


s = socket.socket()         # Create a socket object
host = socket.gethostname() # Get local machine name
port = 12345                # Reserve a port for your service.

s.connect((host, port))

listener_thread = threading.Thread(target=receive_from_socket, args=(s,)).start()

face_data = None

body = bpy.data.objects['Cube']

while True:
    if face_data is not None:
        # Translation matrices
        T0 = T1 = Matrix.Translation((face_data["position"]["x"], face_data["position"]["y"], face_data["position"]["z"]))

        # Rotation matrix
        R0 = Matrix.Rotation(face_data["position"]["roll"], face_data["position"]["pitch"], face_data["position"]["yaw"])

        M = T0 * R0 * T1

        body.matrix_world = M * body.matrix_world

        time.sleep(10)


s.close()