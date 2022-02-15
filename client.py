import socket
import time

sock = socket.socket()         # Create a socket object
host = "127.0.0.1"
port = 12345
sock.connect((host, port))
sock.sendall(b'Hello, world')
data = sock.recv(1024)
print('Received', repr(data))
time.sleep(5)
sock.sendall(b'Hello, world')
data = sock.recv(1024)
print('Received', repr(data))
sock.close()