import tensorflow as tf # type: ignore
from tensorflow.keras import layers # type: ignore
from collections import deque
import random
import numpy as np # type: ignore
import os
import struct
import socket
import threading
import time
from multiprocessing import Pool
from multiprocessing.dummy import Pool as ThreadPool


class Agent:
    def __init__(self, state_size, action_size, model_path=None):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.model = self._build_model()
        self.totalscore = 0
        
        if model_path:
            self.model.load_weights(model_path)
    
    def _build_model(self):
        model = tf.keras.Sequential()
        model.add(layers.Input(shape=(self.state_size + 1,)))
        model.add(layers.Dense(24, activation='relu'))
        model.add(layers.Dense(self.action_size, activation='linear'))
        model.compile()
        return model
    
    def act(self, state):
        out_values = self.model.predict(np.array(state).reshape(1, -1), verbose = None)
        return float(out_values[0][0]) % 360
    
    def remember(self, state, action):
        self.memory.append((state, action))
    
    def save(self, name):
        self.model.save_weights(name)
    
    def load(self, name):
        self.model.load_weights(name)
    
    def get_weights(self):
        return self.model.get_weights()
    
    def set_weights(self, weights):
        self.model.set_weights(weights)

class Manager:
    def __init__(self, num_agents, state_size, action_size, top_k):
        self.num_agents = num_agents
        self.state_size = state_size
        self.action_size = action_size
        self.gen = 1
        self.top_k = top_k  # Number of top agents to retain
        self.agents = [Agent(state_size, action_size) for _ in range(num_agents)]
        self.running = True
        self.lock = threading.Lock()

    def evolve(self):
        print(f'Generation : {self.gen}')
        # Sort agents based on their scores
        self.agents.sort(key=lambda agent: getattr(agent, 'totalscore', 0), reverse=True)
        top_agents = self.agents[:self.top_k]

        # Save the top k models
        for i in range(self.top_k):
            filename = f"top_model_{i + 1}.weights.h5"  # Save each top model with a unique filename
            top_agents[i].save(filename)
            print(f"Top {i + 1} model saved with score: {top_agents[i].totalscore}")

        # Create new generation by selecting from top agents
        new_agents = []
        for i in range(self.top_k):
            new_agents.append(top_agents[i])

        for i in range(self.num_agents - self.top_k):
            parent = random.choice(top_agents)
            child = Agent(self.state_size, self.action_size)
            child.model.set_weights(parent.model.get_weights())
            
            # Mutate the child model by adding small noise
            weights = child.model.get_weights()
            mutated_weights = [w + np.random.normal(0, 0.3, size=w.shape) for w in weights]
            child.model.set_weights(mutated_weights)
            new_agents.append(child)

        self.gen += 1
        self.agents = new_agents

    def save_top(self):
        top_agents = sorted(self.agents, key=lambda agent: getattr(agent, 'totalscore', 0), reverse=True)[:5]
        for i, agent in enumerate(top_agents):
            filename = f"top_model_{i + 1}.weights.h5"
            agent.save(filename)
            print(f"Top {i + 1} model saved with score: {agent.totalscore}")

    def load_top(self):
        for i in range(self.top_k):
            filename = f"top_model_{i + 1}.weights.h5"
            if os.path.exists(filename):
                agent = Agent(self.state_size, self.action_size)
                agent.load(filename)
                self.agents[i] = agent  # Replace the current agent with the loaded top agent
                print(f"Top {i + 1} model loaded from {filename}")

    def reset(self):
        for agent in self.agents:
            # Clear memory or reset any internal variables
            agent.memory = deque(maxlen=2000)
            # Reset score
            agent.totalscore = 0


class UnityServer:
    def __init__(self, host, port, manager):
        self.host = host
        self.port = port
        self.manager = manager
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.bind((self.host, self.port))
        self.server_socket.listen(1)
        print(f"Server listening on {self.host}:{self.port}")
        self.client_socket, self.addr = self.server_socket.accept()
        print(f"Connection from {self.addr}")
        self.thread = threading.Thread(target=self.run)
        self.pool = ThreadPool(4)  # Initialize the pool once
        self.thread.start()

    def predict_single_agent(self, agent, reward, state):
        agent.totalscore += reward
        return agent.act(state)

    def run(self):
        while self.manager.running:
            try:
                data_bytes = self.client_socket.recv(700 * 4)
                if not data_bytes:
                    continue
                data = struct.unpack('700i', data_bytes)
                rewards = data[0:100]
                v_x = data[100:200]
                v_y = data[200:300]
                x = data[300:400]
                y = data[400:500]
                cur_checkpoint = data[500:600]
                is_reset = data[600:700]
                
                # Prepare state for each agent
                states = []
                for i in range(self.manager.num_agents):
                    state = [
                        x[i], y[i],
                        v_x[i], v_y[i],
                        cur_checkpoint[i], 0  # Initial angle is 0
                    ]
                    states.append(state)
                
                # Use pool.map to parallelize agent predictions
                angles = self.pool.starmap(
                    self.predict_single_agent,
                    zip(self.manager.agents, rewards, states)
                )
                
                # Normalize angles to [0, 360)
                angles = [angle % 360 for angle in angles]
                
                # Send actions back to Unity
                angles_bytes = struct.pack('100f', *angles)
                self.client_socket.sendall(angles_bytes)
                
                # Handle resets
                if any(is_reset):
                    with self.manager.lock:
                        print("Evolving agents...")
                        self.manager.evolve()
                        self.manager.reset()
                
            except Exception as e:
                print(f"Error: {e}")
                break
        self.client_socket.close()
        self.server_socket.close()
        self.pool.close()
        self.pool.join()

def main():
    num_agents = 100
    state_size = 5  # Adjust based on your actual state representation (pos_x, pos_y, v_x, v_y, cur_checkpoint)
    action_size = 1   # Assuming single continuous action (angle)
    top_k = 5

    manager = Manager(num_agents, state_size, action_size, top_k)
    manager.load_top()  # Load the top model if available
    server = UnityServer(host='127.0.0.1', port=65432, manager=manager)

    try:
        while True:
            time.sleep(1)  # Keep the main thread alive
    except KeyboardInterrupt:
        print("Stopping training...")
        manager.running = False
        server.thread.join()

if __name__ == "__main__":
    main()