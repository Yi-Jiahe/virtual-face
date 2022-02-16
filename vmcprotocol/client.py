import random

from pythonosc import udp_client


class VmcClient:
    def __init__(self, host, port):
        self.client = udp_client.SimpleUDPClient(host, port)

    def examples(self):
        self.client.send_message("/VMC/Ext/Set/Eye",
                            [1, random.uniform(-1, 1), random.uniform(-1, 1), random.uniform(-1, 1)])

        self.client.send_message("/VMC/Ext/Set/Calib/Ready", "")
        self.client.send_message("/VMC/Ext/Set/Calib/Exec", 0)
        self.client.send_message("/VMC/Ext/Hmd/Pos",
                                 ["head", 0.5, 0.5, 0.8, 0.0, 0.0, 0.0, 0.0])
        self.client.send_message("/VMC/Ext/Tra/Pos",
                                 ["tra1", random.uniform(-0.1,0.1), random.uniform(-0.1,0.1), random.uniform(-0.1,0.3), 0.0, 0.0, 0.0, 0.0])
        self.client.send_message("/VMC/Ext/Tra/Pos",
                                 ["tra2", random.uniform(-0.1,0.1), random.uniform(-0.1,0.1), random.uniform(-0.1,0.3), 0.0, 0.0, 0.0, 0.0])