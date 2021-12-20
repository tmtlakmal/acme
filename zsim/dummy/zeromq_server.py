import zmq
import json


class ZeroMQServer:

    def __init__(self, port="tcp://*:5555"):
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REP)
        self.socket.bind(port)

    def receive(self):
        message = self.socket.recv()
        return json.loads(message.decode("utf-8"))

    def send(self, message_to_send):
        json_string = json.dumps(message_to_send)
        self.socket.send(bytearray(json_string, encoding="utf-8"))

    def send_and_receive(self, message_to_send):
        json_string = json.dumps(message_to_send)
        self.socket.send(bytearray(json_string, encoding="utf-8"))
        # wait for the reply
        message = self.socket.recv()
        return json.loads(message.decode('utf-8'))
