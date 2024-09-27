import tensorflow as tf # type: ignore
from tensorflow.keras import layers # type: ignore
import tensorflow.keras.backend as K # type: ignore
import random
import numpy as np # type: ignore
import os
import struct
import socket
import threading
import time
from multiprocessing.dummy import Pool #Remove .dummy and at Pool() -> Pool(Process = 4)
import gc

# -------------0---1----2----3----4-----5-----6----7----8-----9-----10-----11-----12--
gen_changes = [1, 0.5, 0.3, 0.3, 0.3, 0.2, 0.1, 0.05, 0.03, 0.03, 0.02, 0.01, 0.001]
global hcheckpoint 
variable_path = 'variable.txt'
if os.path.exists(variable_path):
    file1 = open(variable_path,"r+")
    hcheckpoint = int(file1.read())
else:
    hcheckpoint = 0
    file1 = open(variable_path,"w")
    file1.write(str(hcheckpoint))
file1.close()
gen_change = gen_changes[hcheckpoint]
print(f"hcheckpoint : {hcheckpoint}")

class Agent:
    def __init__(self, state_size, action_size, model_path=None):
        self.state_size = state_size
        self.action_size = action_size
        self.model = self._build_model()
        self.totalscore = 0
        self.checkpoint = 0
        
        if model_path:
            self.model.load_weights(model_path)
    
    def _build_model(self):
        model = tf.keras.Sequential()
        model.add(layers.Input(shape=(self.state_size,)))
        model.add(layers.Dense(64, activation='relu'))
        model.add(layers.Dense(64, activation='relu'))
        model.add(layers.Dense(self.action_size, activation='linear'))
        return model
    
    def save(self, name):
        self.model.save_weights(name)
    
    def load(self, name):
        self.model.load_weights(name)

# @tf.function(reduce_retracing=True, experimental_relax_shapes=True)
def act(agent, state):
    # out_values = agent.model(np.array(state).reshape(1, -1), training=False)
    out_values = agent.model.predict(np.array(state).reshape(1, -1), verbose=False)
    return float(out_values[0][0]) % 360


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
        self.checkpoint = 0

    def evolve(self):
        global hcheckpoint
        
        print(f'Generation : {self.gen}')
        # Sort agents based on their scores
        self.agents.sort(key=lambda agent: getattr(agent, 'totalscore', 0), reverse=True)
        top_agents = self.agents[:self.top_k]

        for agent in top_agents:
            if agent.checkpoint > hcheckpoint:
                hcheckpoint = agent.checkpoint
                filename = f"Archive_model/checkpoint_{hcheckpoint}.weights.h5"
                agent.save(filename)
        

        # Save the top k models
        for i in range(self.top_k):
            filename = f"top_model_{i + 1}.weights.h5"  # Save each top model with a unique filename
            top_agents[i].save(filename)
            self.agents[i] = top_agents[i]
            print(f"Top {i + 1} model saved with score: {top_agents[i].totalscore}, checkpoint: {top_agents[i].checkpoint}")


        gen_change = gen_changes[min(hcheckpoint, 12)]
        print("Evolving agents...")
        print(f'Mutate factor = {gen_change}')
        del self.agents
        gc.collect()
        self.agents = []

        for parent in top_agents:
            child = Agent(self.state_size, self.action_size)
            child.model.set_weights(parent.model.get_weights())
            self.agents.append(child)

        for i in range(self.num_agents - self.top_k):
            parent = random.choice(top_agents)
            child = Agent(self.state_size, self.action_size)
            child.model.set_weights(parent.model.get_weights())
            # Mutate the child model by adding small noise
            weights = child.model.get_weights()
            mutated_weights = [w + np.random.normal(0, gen_change, size=w.shape) for w in weights]
            child.model.set_weights(mutated_weights)
            self.agents.append(child)


        del top_agents
        gc.collect()

        self.gen += 1

    def load_top(self):
        have = 0
        top_agents = []
        del self.agents
        gc.collect()
        self.agents = []
        for i in range(self.top_k):
            filename = f"top_model_{i + 1}.weights.h5"
            if os.path.exists(filename):
                agent = Agent(self.state_size, self.action_size)
                agent.load(filename)
                top_agents.append(agent)  # Replace the current agent with the loaded top agent
                print(f"Top {i + 1} model loaded from {filename}")
                have += 1

        for parent in top_agents:
            child = Agent(self.state_size, self.action_size)
            child.model.set_weights(parent.model.get_weights())
            self.agents.append(child)

        for i in range(self.num_agents - len(top_agents)):
            parent = random.choice(top_agents)
            child = Agent(self.state_size, self.action_size)
            child.model.set_weights(parent.model.get_weights())
            # Mutate the child model by adding small noise
            weights = child.model.get_weights()
            mutated_weights = [w + np.random.normal(0, gen_change, size=w.shape) for w in weights]
            child.model.set_weights(mutated_weights)
            self.agents.append(child)

                

    def reset(self):
        for agent in self.agents:
            agent.totalscore = 0
            agent.checkpoint = 0


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
        self.pool = Pool(processes = 100)
        self.thread.start()

    def run(self):
        count = 0
        while self.manager.running:
            try:
                data_bytes = self.client_socket.recv(801 * 4)
                if not data_bytes:
                    continue
                data = struct.unpack('801i', data_bytes)
                rewards = data[0:100]
                v_x = data[100:200]
                v_y = data[200:300]
                x = data[300:400]
                y = data[400:500]
                checkpoint_x = data[500:600]
                checkpoint_y = data[600:700]
                cur_checkpoint = data[700:800]
                is_reset = data[800]
                
                # Prepare state for each agent
                states = []
                for i in range(self.manager.num_agents):
                    state = [
                        x[i], y[i],
                        v_x[i], v_y[i], 
                        checkpoint_x[i], checkpoint_y[i],
                        cur_checkpoint[i]
                    ]
                    states.append(state)
                

                t1 = time.time()
                angles = self.pool.starmap(act, zip(self.manager.agents, states))
                # print(f'Finish with : {time.time() - t1}')

                for i, reward in enumerate(rewards):
                    self.manager.agents[i].totalscore += reward
                    self.manager.agents[i].checkpoint = max(cur_checkpoint[i], self.manager.agents[i].checkpoint)

                angles.extend([0] * (100 - self.manager.num_agents))
                
                # Send actions back to Unity
                angles_bytes = struct.pack('100f', *angles)
                self.client_socket.sendall(angles_bytes)

                # print(count)
                # if count % 10 == 9:
                # K.clear_session()
                # gc.collect()
                count += 1

                # Handle resets
                if is_reset:
                    with self.manager.lock:
                        self.manager.evolve()
                        self.manager.reset()
                
            except Exception as e:
                print(f"Error: {e}")
                break
        self.client_socket.close()
        self.server_socket.close()

def main():
    num_agents = 100
    state_size = 7  # Adjust based on your actual state representation (pos_x, pos_y, v_x, v_y, checkpoint_x, checkpoint_y, cur_checkpoint)
    action_size = 1   # Assuming single continuous action (angle)
    top_k = 5

    manager = Manager(num_agents, state_size, action_size, top_k)
    manager.load_top()  # Load the top model if available
    server = UnityServer(host='127.0.0.1', port=65432, manager=manager)


    try:
        while True:
            time.sleep(150)  # Keep the main thread alive
    except KeyboardInterrupt:
        print("Stopping training...")
        file1 = open(variable_path,"w")
        file1.write(str(hcheckpoint))
        file1.close()
        manager.running = False
        server.thread.join()
        server.pool.close()
        server.pool.join()

if __name__ == "__main__":
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print("Memory growth enabled for GPUs")

            # tf.config.set_logical_device_configuration(
            #     gpus[0],
            #     [tf.config.LogicalDeviceConfiguration(memory_limit=4096)])  # Set limit to 4096 MB (4GB)
            # print("Memory limit set to 4GB")
        except RuntimeError as e:
            print(f"Error enabling memory growth: {e}")
    else:
        print("No GPU devices found.")
    

    main()