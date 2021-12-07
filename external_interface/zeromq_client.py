import zmq
import json

from zsim.envs.external_messages import TrafficData


class ZeroMqClient():


    def __init__(self, port="tcp://localhost:5555"):
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REQ)
        self.socket.connect(port)

    def send_message(self, message_to_send):

        message = self.build_message(message_to_send)
        self.socket.send(message)

        #wait for the reply
        message = self.socket.recv()
        return json.loads(message.decode('utf-8'))

    def build_message(self, message):
        json_string = json.dumps(message, default=lambda o: o.__dict__)
        byte_string = bytearray(json_string, encoding="utf-8")
        return byte_string