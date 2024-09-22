import tensorflow as tf
from tensorflow.keras import layers
from collections import deque
import random
import numpy as np
import os
import struct
import socket
import threading
import time

class Agent:
    def __init__(self, state_size, action_size, model_path=None):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.model = self._build_model()
        self.score = 0
        
        if model_path:
            self.model.load_weights(model_path)
    
    def _build_model(self):
        model = tf.keras.Sequential()
        model.add(layers.Input(shape=(self.state_size,)))
        model.add(layers.Dense(64, input_dim=self.state_size, activation='relu'))
        model.add(layers.Dense(64, activation='relu'))
        model.add(layers.Dense(self.action_size, activation='linear'))
        model.compile(loss='mse')
        return model
    
    def act(self, state):
        out_values = self.model.predict(state, verbose=0)
        return float(out_values[0][0])  # Assuming action_size=1 for angle
    
    def remember(self, state, action, reward, next_state):
        self.memory.append((state, action, reward, next_state))
    
    def replay(self, batch_size=64):
        if len(self.memory) < batch_size:
            return
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state in minibatch:
            target = reward
            target_f = self.model.predict(state, verbose=0)
            target_f[0][0] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
    
    def save(self, name):
        self.model.save_weights(name)
    
    def load(self, name):
        self.model.load_weights(name)
    
    def get_weights(self):
        return self.model.get_weights()
    
    def set_weights(self, weights):
        self.model.set_weights(weights)

class Manager:
    def __init__(self, num_agents=100, state_size=600, action_size=1, top_k=10):
        self.num_agents = num_agents
        self.state_size = state_size
        self.action_size = action_size
        self.top_k = top_k  # Number of top agents to retain
        self.agents = [Agent(state_size, action_size) for _ in range(num_agents)]
        self.best_agent = None
        self.best_score = -float('inf')
        self.running = True
        self.lock = threading.Lock()

    def evolve(self):
        # Sort agents based on their scores
        self.agents.sort(key=lambda agent: getattr(agent, 'score', 0), reverse=True)
        top_agents = self.agents[:self.top_k]

        # Update the best agent
        if top_agents[0].score > self.best_score:
            self.best_score = top_agents[0].score
            self.best_agent = top_agents[0]
            print(f"New best score: {self.best_score}")

        # Save the top k models
        for i in range(self.top_k):
            filename = f"top_model_{i + 1}.weights.h5"  # Save each top model with a unique filename
            top_agents[i].save(filename)
            print(f"Top {i + 1} model saved with score: {top_agents[i].score}")

        # Create new generation by selecting from top agents
        new_agents = []
        for i in range(self.num_agents):
            parent = random.choice(top_agents)
            child = Agent(self.state_size, self.action_size)
            child.model.set_weights(parent.model.get_weights())
            
            # Mutate the child model by adding small noise
            weights = child.model.get_weights()
            mutated_weights = [w + np.random.normal(0, 0.01, size=w.shape) for w in weights]
            child.model.set_weights(mutated_weights)
            new_agents.append(child)

        self.agents = new_agents

    def save_top(self):
        top_agents = sorted(self.agents, key=lambda agent: getattr(agent, 'score', 0), reverse=True)[:5]
        for i, agent in enumerate(top_agents):
            filename = f"top_model_{i + 1}.weights.h5"
            agent.save(filename)
            print(f"Top {i + 1} model saved with score: {agent.score}")

    def load_top(self):
        for i in range(5):
            filename = f"top_model_{i + 1}.weights.h5"
            if os.path.exists(filename):
                agent = Agent(self.state_size, self.action_size)
                agent.load(filename)
                self.agents[i] = agent  # Replace the current agent with the loaded top agent
                print(f"Top {i + 1} model loaded from {filename}")

    def reset(self):
        # Clear memory or reset any internal variables
        self.memory.clear()
        # Reset score if necessary
        self.score = 0


class UnityServer:
    def __init__(self, host='127.0.0.1', port=65432, manager=None):
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
        self.thread.start()

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
                        cur_checkpoint[i],
                    ]
                    states.append(np.array(state).reshape(1, -1))
                
                # Get actions from agents
                angles = []
                for i, agent in enumerate(self.manager.agents):
                    agent.score += rewards[i]
                    angle = agent.act(states[i])
                    angles.append(angle)
                
                # Send actions back to Unity
                angles_bytes = struct.pack('100f', *angles)
                self.client_socket.sendall(angles_bytes)
                
                # Store rewards and other info for training
                for i, agent in enumerate(self.manager.agents):
                    reward = rewards[i]
                    agent.remember(states[i], angles[i], reward, states[i])
                    # Update agent's score for evolutionary selection
                    if not hasattr(agent, 'score'):
                        agent.score = 0
                    agent.score += reward  # Customize based on your scoring system

                for i, agent in enumerate(self.manager.agents):
                    if is_reset[i] == 1:
                        with self.manager.lock:
                            print("Replaying and evolving agents...")
                            for agent in self.manager.agents:
                                agent.replay()
                            self.manager.evolve()

            except Exception as e:
                print(f"Error: {e}")
                break
        self.client_socket.close()
        self.server_socket.close()

def main():
    num_agents = 100
    state_size = 5  # Adjust based on your actual state representation
    action_size = 1    # Assuming single continuous action (angle)
    top_k = 10

    manager = Manager(num_agents=num_agents, state_size=state_size, action_size=action_size, top_k=top_k)
    manager.load_top()  # Load the top model if available
    server = UnityServer(host='127.0.0.1', port=65432, manager=manager)

    try:
        while True:
            time.sleep(1)  # Keep the main thread alive
    except KeyboardInterrupt:
        print("Stopping training...")
        manager.save_top()
        manager.running = False
        server.thread.join()

if __name__ == "__main__":
    main()