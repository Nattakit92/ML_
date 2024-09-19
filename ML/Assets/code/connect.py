import socket
import struct
import random

# Create a TCP/IP socket
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

# Bind the socket to a local address and port
host = '127.0.0.1'  # Localhost
port = 65432        # Port to listen on
server_socket.bind((host, port))

# Listen for incoming connections
server_socket.listen(1)
print(f'Server listening on {host}:{port}')

# Wait for a connection
connection, client_address = server_socket.accept()
print(f'Connected to: {client_address}')

def RL(scores): #Return angle (0,360)
    out = random.uniform(0, 1)
    return out * 360

while True:
    # Receive data (100 floats = 100 * 4 bytes)
    data = connection.recv(100 * 4)
    if data:
        # Unpack the 100 floats from the received data
        scores = struct.unpack('100f', data)

        reset = all(score == 0 for score in scores)
        
        if reset:
            print("helloworld")

        # Prepare a response: a random array of 100 floats in the range 0-360
        response_array = [RL(scores) for _ in range(100)]
        response_bytes = struct.pack('100f', *response_array)

        # Send the response back to the client
        connection.sendall(response_bytes)
    else:
        break

# Close the connection
connection.close()
server_socket.close()
