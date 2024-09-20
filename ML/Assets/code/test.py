import socket
import struct
import random
import threading
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from collections import deque

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

def build_model(state_size, action_size):
    model = tf.keras.Sequential()
    model.add(layers.Dense(24, input_dim=state_size, activation='relu'))
    model.add(layers.Dense(24, activation='relu'))
    model.add(layers.Dense(action_size, activation='linear'))
    model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(learning_rate=0.001))
    return model

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95  # Discount factor
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = build_model(state_size, action_size)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = reward + self.gamma * np.amax(self.model.predict(next_state)[0])
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)

def RL(reward, v_x, v_y, x, y, cur_checkpoint, state, agent, done): # Return angle (0,360)
    next_state = np.array([v_x, v_y, x, y, cur_checkpoint]).reshape(1, -1)  # Reshaping for agent input
    action = agent.act(state)
    agent.remember(state, action, reward, next_state, done)
    return action * 20, next_state  # Action * 10 gives the angle between 0 and 360

def train_agent(agent, batch_size):
    while True:
        if len(agent.memory) > batch_size:
            agent.replay(batch_size)


agent = DQNAgent(5, 18)
batch_size = 256
# Start training in a separate thread
training_thread = threading.Thread(target=train_agent, args=(agent, batch_size))
training_thread.daemon = True
training_thread.start()

# Initialize states for all agents
states = [np.zeros((1, 5)) for _ in range(100)]
count = 0;

while count <= 900:
    data_bytes = connection.recv(700 * 4)
    score_avg = 0
    if data_bytes:
        data = struct.unpack('700i', data_bytes)
        rewards = data[0:100]
        v_x = data[100:200]
        v_y = data[200:300]
        x = data[300:400]
        y = data[400:500]
        cur_checkpoint = data[500:600]
        done = data[600:700]
        
        if done == 0:
            states = []
            for i in range(100):
                state = np.array([v_x[i], v_y[i], x[i], y[i], cur_checkpoint[i]]).reshape(1,5)
                states.append(state)
            count+=1
            print(count)
            done = 1

        angles = []
        new_states = []
        
        # Process each agent (vehicle)
        for i in range(100):
            score_avg += rewards[i]
            angle, next_state = RL(rewards[i], v_x[i], v_y[i], x[i], y[i], cur_checkpoint[i], states[i], agent, done[i])
            angles.append(angle)
            new_states.append(next_state)

        # Convert angles list to bytes and send back to the client
        angles_bytes = struct.pack('100f', *angles)
        connection.sendall(angles_bytes)


        # Update states for all agents
        states = new_states

        print(score_avg / 100)

    else:
        break

# Save the entire model to a file
agent.model.save('dqn_model.h5')

# Close the connection
connection.close()
server_socket.close()
