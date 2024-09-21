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

class DQNAgent:
    def __init__(self, state_size, action_size, model_path=None):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self._build_model()
        
        if model_path:
            self.model.load_weights(model_path)
    
    def _build_model(self):
        model = tf.keras.Sequential()
        model.add(layers.Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(layers.Dense(24, activation='relu'))
        model.add(layers.Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate))
        return model
    
    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return np.random.uniform(0, 360)  # Continuous action space
        q_values = self.model.predict(state, verbose=0)
        return float(q_values[0][0])  # Assuming action_size=1 for angle
    
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    
    def replay(self, batch_size=64):
        if len(self.memory) < batch_size:
            return
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target += self.gamma * np.amax(self.model.predict(next_state, verbose=0)[0])
            target_f = self.model.predict(state, verbose=0)
            target_f[0][0] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def save(self, name):
        self.model.save_weights(name)
    
    def load(self, name):
        self.model.load_weights(name)
    
    def get_weights(self):
        return self.model.get_weights()
    
    def set_weights(self, weights):
        self.model.set_weights(weights)

class DQNManager:
    def __init__(self, num_agents=100, state_size=600, action_size=1, top_k=10):
        self.num_agents = num_agents
        self.state_size = state_size
        self.action_size = action_size
        self.top_k = top_k  # Number of top agents to retain
        self.agents = [DQNAgent(state_size, action_size) for _ in range(num_agents)]
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
            self.best_agent.save("best_model.h5")
            print(f"New best score: {self.best_score}")

        # Create new generation
        new_agents = []
        for i in range(self.num_agents):
            # Select a top agent
            parent = random.choice(top_agents)
            child = DQNAgent(self.state_size, self.action_size)
            child.model.set_weights(parent.model.get_weights())
            
            # Mutate the child model by adding small noise
            weights = child.model.get_weights()
            mutated_weights = [w + np.random.normal(0, 0.01, size=w.shape) for w in weights]
            child.model.set_weights(mutated_weights)
            new_agents.append(child)
        self.agents = new_agents

    def reset(self):
        # Clear memory or reset any internal variables
        self.memory.clear()
        # Optionally reset epsilon or other parameters
        self.epsilon = 1.0
        # Reset score if necessary
        self.score = 0

    def save_best(self):
        if self.best_agent:
            self.best_agent.save("best_model.h5")
            print("Best model saved.")

    def load_best(self):
        if os.path.exists("best_model.h5"):
            self.best_agent = DQNAgent(self.state_size, self.action_size)
            self.best_agent.load("best_model.h5")
            print("Best model loaded.")

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
                done = data[600:700]
                
                # Prepare state for each agent
                states = []
                for i in range(self.manager.num_agents):
                    state = [
                        x[i], y[i],
                        v_x[i], v_y[i],
                        cur_checkpoint[i],
                        rewards[i]
                    ]
                    states.append(np.array(state).reshape(1, -1))
                
                # Get actions from agents
                angles = []
                for i, agent in enumerate(self.manager.agents):
                    angle = agent.act(states[i])
                    angles.append(angle)
                
                # Send actions back to Unity
                angles_bytes = struct.pack('100f', *angles)
                self.client_socket.sendall(angles_bytes)
                
                # Store rewards and other info for training
                for i, agent in enumerate(self.manager.agents):
                    reward = rewards[i]
                    agent.remember(states[i], angles[i], reward, states[i], done[i])
                    # Update agent's score for evolutionary selection
                    if not hasattr(agent, 'score'):
                        agent.score = 0
                    agent.score += reward  # Customize based on your scoring system

                if done == 1:
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
    state_size = 6  # Adjust based on your actual state representation
    action_size = 1    # Assuming single continuous action (angle)
    top_k = 10

    manager = DQNManager(num_agents=num_agents, state_size=state_size, action_size=action_size, top_k=top_k)
    manager.load_best()  # Load the best model if available
    server = UnityServer(host='127.0.0.1', port=65432, manager=manager)

    try:
        while True:
            time.sleep(1)  # Keep the main thread alive
    except KeyboardInterrupt:
        print("Stopping training...")
        manager.save_best()
        manager.running = False
        server.thread.join()

if __name__ == "__main__":
    main()