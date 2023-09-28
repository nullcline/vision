import argparse
from plane import Plane
from ground import Server
import socket

ip_adr = socket.gethostbyname(socket.gethostname())
print(f"Detected default IPv4 Address: {ip_adr}")
ap = argparse.ArgumentParser()
ap.add_argument(
    "-s",
    "--server-ip",
    required=False,
    default=ip_adr,
    help="IP address of the server to which the client will connect",
)
args = vars(ap.parse_args())

client = Plane(args["server_ip"])
server = Server()

client.start()
server.start()
