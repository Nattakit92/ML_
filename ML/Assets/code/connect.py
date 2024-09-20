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

def RL(reward, v_x, v_y, x, y): #Return angle (0,360)
    out = random.uniform(0, 1)
    return out * 360

while True:
    data_bytes = connection.recv(700 * 4)
    if data_bytes:
        # Unpack the 100 floats from the received data
        data = struct.unpack('700i', data_bytes)
        rewards = data[0:100]
        v_x = data[100:200]
        v_y = data[200:300]
        x = data[300:400]
        y = data[400:500]
        cur_checkpoint = data[500:600]
        done = data[600:700]
        
        print(done);

        # Prepare a response: a random array of 100 floats in the range 0-360
        angles = [RL(rewards[i], v_x[i], v_y[i], x[i], y[i]) for i in range(100)]
        angles_bytes = struct.pack('100f', *angles)

        # Send the response back to the client
        connection.sendall(angles_bytes)
    else:
        break

# Close the connection
connection.close()
server_socket.close()
