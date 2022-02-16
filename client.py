import socket


class Client:
    def __init__(self, host, port):
        self.addr = (host, port)
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    # TODO: Find a way to break out of this loop
    def connect(self):
        while True:
            try:
                self.sock.connect(self.addr)
                print(f"Connected to {self.addr[0], self.addr[1]}")
                break
            except ConnectionRefusedError:
                print(f"Failed to connect to {self.addr[0], self.addr[1]}")
                print("Retrying...")

    def send(self, b):
        self.sock.sendall(b)

    def close_socket(self):
        self.sock.close()

