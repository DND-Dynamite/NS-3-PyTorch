# nano_network_simulator.py
import simpy
import numpy as np
import torch
import torch.nn as nn
import random
import matplotlib.pyplot as plt
from collections import deque, defaultdict
import json
import time
from datetime import datetime

class THzChannel:
    """Simulates THz channel characteristics for nano networks"""
    def __init__(self, center_frequency=1.0):  # 1.0 THz
        self.center_frequency = center_frequency
        self.molecular_absorption = {
            'H2O': 0.1,  # Water vapor absorption
            'O2': 0.05,  # Oxygen absorption
            'CO2': 0.02  # CO2 absorption
        }
        self.spreading_loss = 2.0  # Path loss exponent
        self.noise_floor = -90  # dBm
        
    def calculate_path_loss(self, distance):
        """Calculate THz path loss with molecular absorption"""
        spreading_loss = 20 * np.log10(distance) + 20 * np.log10(self.center_frequency) + 92.45
        molecular_absorption = sum(self.molecular_absorption.values()) * distance
        return spreading_loss + molecular_absorption
    
    def calculate_snr(self, tx_power, distance):
        """Calculate Signal-to-Noise Ratio"""
        path_loss = self.calculate_path_loss(distance)
        received_power = tx_power - path_loss
        return received_power - self.noise_floor

class NanoNode:
    """Represents a nano node in the network"""
    def __init__(self, node_id, env, channel, x=0, y=0, is_malicious=False):
        self.node_id = node_id
        self.env = env
        self.channel = channel
        self.position = (x, y)
        self.is_malicious = is_malicious
        self.packet_queue = deque()
        self.transmission_power = -20  # dBm for nano devices
        self.data_rate = 100e3  # 100 kbps
        self.neighbors = []
        self.sent_packets = 0
        self.received_packets = 0
        
    def add_neighbor(self, neighbor_node):
        """Add a neighbor node"""
        if neighbor_node != self:
            self.neighbors.append(neighbor_node)
    
    def calculate_distance(self, other_node):
        """Calculate distance to another node"""
        x1, y1 = self.position
        x2, y2 = other_node.position
        return np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    
    def send_packet(self, packet_size=512, destination=None):
        """Send a packet to destination"""
        if destination is None and self.neighbors:
            destination = random.choice(self.neighbors)
        
        if destination:
            packet = {
                'source': self.node_id,
                'destination': destination.node_id,
                'size': packet_size,
                'timestamp': self.env.now,
                'is_malicious': self.is_malicious
            }
            
            # Malicious nodes send at much higher rates
            if self.is_malicious:
                packet_size = random.randint(1024, 2048)  # Larger packets
                
            self.sent_packets += 1
            destination.receive_packet(packet, self)
    
    def receive_packet(self, packet, sender):
        """Receive a packet from sender"""
        distance = self.calculate_distance(sender)
        snr = self.channel.calculate_snr(sender.transmission_power, distance)
        
        # Packet success based on SNR
        if snr > 10:  # SNR threshold for successful reception
            self.received_packets += 1
            self.packet_queue.append(packet)
            return True
        return False
    
    def normal_behavior(self):
        """Normal node behavior - periodic communication"""
        while True:
            # Normal nodes send packets periodically
            if not self.is_malicious:
                yield self.env.timeout(random.expovariate(1.0))  # ~1 packet per second
                if self.neighbors:
                    self.send_packet()
            else:
                # Malicious nodes wait before attacking
                yield self.env.timeout(15)  # Start attack after 15 seconds
                break
    
    def dos_attack_behavior(self):
        """Malicious node behavior - DoS attack"""
        while True:
            # Flood the network with packets
            for neighbor in self.neighbors:
                for _ in range(10):  # Send multiple packets
                    self.send_packet(packet_size=2048, destination=neighbor)
            yield self.env.timeout(random.expovariate(10.0))  # High rate

class DoSDetector(nn.Module):
    """PyTorch model for DoS detection in nano networks"""
    def __init__(self, input_features=8):
        super(DoSDetector, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_features, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 2),
            nn.Softmax(dim=1)
        )
        
    def forward(self, x):
        return self.network(x)

class AIDoSDetectionSystem:
    """AI-powered DoS detection and mitigation system"""
    def __init__(self, env):
        self.env = env
        self.model = DoSDetector()
        self.detector = RealTimeDetector()
        self.traffic_history = defaultdict(lambda: deque(maxlen=100))
        self.attack_log = []
        self.mitigation_actions = []
        
        # Load or initialize model
        try:
            self.model.load_state_dict(torch.load('nano_dos_model.pth'))
            print("AI DoS detection model loaded")
        except:
            print("Initializing new DoS detection model")
    
    def extract_traffic_features(self, node, time_window=10):
        """Extract traffic features for ML analysis"""
        current_time = self.env.now
        node_history = self.traffic_history[node.node_id]
        
        # Filter recent traffic
        recent_packets = [p for p in node_history 
                         if current_time - p['timestamp'] <= time_window]
        
        if len(recent_packets) < 5:
            return None
        
        # Calculate features
        packet_sizes = [p['size'] for p in recent_packets]
        packet_rates = len(recent_packets) / time_window
        
        features = np.array([
            packet_rates,                           # Packet rate
            np.mean(packet_sizes),                  # Avg packet size
            np.std(packet_sizes) if len(packet_sizes) > 1 else 0,  # Size variation
            np.max(packet_sizes),                   # Max packet size
            len([p for p in recent_packets if p['size'] > 1024]),  # Large packets
            node.sent_packets / (current_time + 1), # Overall send rate
            len(recent_packets) / (len(node_history) + 1),  # Traffic burstiness
            random.random()  # Placeholder for error rate
        ], dtype=np.float32)
        
        return features
    
    def detect_dos(self, node):
        """Detect DoS attacks using AI/ML"""
        features = self.extract_traffic_features(node)
        if features is None:
            return False, 0.0
        
        # Use PyTorch model for detection
        with torch.no_grad():
            input_tensor = torch.FloatTensor(features).unsqueeze(0)
            output = self.model(input_tensor)
            attack_prob = output[0][1].item()
            
            is_attack = attack_prob > 0.7
            if is_attack:
                self.log_attack(node, attack_prob, features)
                
            return is_attack, attack_prob
    
    def log_attack(self, node, probability, features):
        """Log detected attack"""
        attack_entry = {
            'timestamp': self.env.now,
            'node_id': node.node_id,
            'probability': probability,
            'features': features.tolist(),
            'position': node.position
        }
        self.attack_log.append(attack_entry)
        print(f"🚨 DoS ATTACK DETECTED - Node {node.node_id} "
              f"at time {self.env.now:.2f}s (prob: {probability:.3f})")
    
    def mitigate_attack(self, node):
        """Execute mitigation strategies"""
        mitigation = {
            'timestamp': self.env.now,
            'node_id': node.node_id,
            'actions': [
                f"Rate limiting node {node.node_id}",
                f"Isolating node {node.node_id} from network",
                f"Alerting network administrator"
            ]
        }
        self.mitigation_actions.append(mitigation)
        
        print(f"🛡️ MITIGATION ACTIONS for Node {node.node_id}:")
        for action in mitigation['actions']:
            print(f"  → {action}")
        
        # In simulation, we can disable malicious nodes
        node.neighbors = []  # Isolate the node
    
    def monitor_traffic(self, packet, source_node):
        """Monitor network traffic for analysis"""
        self.traffic_history[source_node.node_id].append(packet)

class RealTimeDetector:
    """Lightweight real-time detector for continuous monitoring"""
    def __init__(self):
        self.packet_threshold = 50  # packets/second
        self.size_threshold = 1024  # bytes
        self.anomaly_score = 0
        
    def update_detection(self, packet_rate, avg_packet_size):
        """Update anomaly detection score"""
        if packet_rate > self.packet_threshold:
            self.anomaly_score += 1
        if avg_packet_size > self.size_threshold:
            self.anomaly_score += 0.5
            
        # Decay anomaly score over time
        self.anomaly_score *= 0.95
        
        return self.anomaly_score > 5.0

class NanoNetworkSimulation:
    """Main simulation class"""
    def __init__(self, num_normal_nodes=20, num_malicious_nodes=3, simulation_time=60):
        self.env = simpy.Environment()
        self.channel = THzChannel()
        self.nodes = []
        self.simulation_time = simulation_time
        self.detection_system = AIDoSDetectionSystem(self.env)
        
        # Create network topology
        self.create_network(num_normal_nodes, num_malicious_nodes)
        
    def create_network(self, num_normal, num_malicious):
        """Create nano network with normal and malicious nodes"""
        # Create normal nodes
        for i in range(num_normal):
            x, y = np.random.rand(2) * 100  # Random position in 100x100 area
            node = NanoNode(f"Normal_{i}", self.env, self.channel, x, y)
            self.nodes.append(node)
        
        # Create malicious nodes
        for i in range(num_malicious):
            x, y = np.random.rand(2) * 100
            node = NanoNode(f"Malicious_{i}", self.env, self.channel, x, y, is_malicious=True)
            self.nodes.append(node)
        
        # Create random network connections
        self.create_network_topology()
    
    def create_network_topology(self):
        """Create random network connections based on distance"""
        for i, node1 in enumerate(self.nodes):
            for j, node2 in enumerate(self.nodes):
                if i != j:
                    distance = node1.calculate_distance(node2)
                    if distance < 30:  #Communication range
                        node1.add_neighbor(node2)
    
    def run_simulation(self):
        """Run the complete simulation"""
        print("Starting Nano Network Simulation with AI/ML DoS Detection...")
        print(f"Network: {len([n for n in self.nodes if not n.is_malicious])} normal nodes, "
              f"{len([n for n in self.nodes if n.is_malicious])} malicious nodes")
        
        # Start node behaviors
        for node in self.nodes:
            if node.is_malicious:
                self.env.process(node.normal_behavior())
                self.env.process(node.dos_attack_behavior())
            else:
                self.env.process(node.normal_behavior())
        
        # Start monitoring process
        self.env.process(self.monitor_network())
        
        # Run simulation
        start_time = time.time()
        self.env.run(until=self.simulation_time)
        simulation_duration = time.time() - start_time
        
        print(f"\n=== SIMULATION COMPLETED ===")
        print(f"Real time: {simulation_duration:.2f}s, Simulation time: {self.simulation_time}s")
        self.analyze_results()
    
    def monitor_network(self):
        """Continuous network monitoring for DoS detection"""
        while True:
            # Check each node for suspicious activity
            for node in self.nodes:
                is_attack, probability = self.detection_system.detect_dos(node)
                
                if is_attack and node.is_malicious:
                    self.detection_system.mitigate_attack(node)
            
            yield self.env.timeout(2.0)  # Check every 2 simulation seconds
    
    def analyze_results(self):
        """Analyze and visualize simulation results"""
        print("\n=== NETWORK ANALYSIS ===")
        
        total_packets = sum(node.sent_packets for node in self.nodes)
        successful_packets = sum(node.received_packets for node in self.nodes)
        
        print(f"Total packets sent: {total_packets}")
        print(f"Successful deliveries: {successful_packets}")
        print(f"Delivery ratio: {successful_packets/total_packets*100:.2f}%")
        print(f"DoS attacks detected: {len(self.detection_system.attack_log)}")
        print(f"Mitigation actions taken: {len(self.detection_system.mitigation_actions)}")
        
        # Visualize results
        self.visualize_network()
        self.plot_detection_timeline()
    
    def visualize_network(self):
        """Visualize the network topology and attacks"""
        plt.figure(figsize=(15, 5))
        
        # Plot 1: Network topology
        plt.subplot(131)
        for node in self.nodes:
            color = 'red' if node.is_malicious else 'blue'
            marker = 'x' if node.is_malicious else 'o'
            plt.scatter(node.position[0], node.position[1], c=color, marker=marker, s=100)
            plt.text(node.position[0], node.position[1], node.node_id.split('_')[1], fontsize=8)
        
        plt.title('Nano Network Topology')
        plt.xlabel('X position (m)')
        plt.ylabel('Y position (m)')
        plt.grid(True, alpha=0.3)
        
        # Plot 2: Traffic analysis
        plt.subplot(132)
        normal_nodes = [n for n in self.nodes if not n.is_malicious]
        malicious_nodes = [n for n in self.nodes if n.is_malicious]
        
        normal_traffic = [n.sent_packets for n in normal_nodes]
        malicious_traffic = [n.sent_packets for n in malicious_nodes]
        
        categories = ['Normal Nodes', 'Malicious Nodes']
        traffic_values = [sum(normal_traffic), sum(malicious_traffic)]
        
        plt.bar(categories, traffic_values, color=['blue', 'red'])
        plt.title('Network Traffic Comparison')
        plt.ylabel('Packets Sent')
        
        # Plot 3: Detection timeline
        plt.subplot(133)
        if self.detection_system.attack_log:
            detection_times = [log['timestamp'] for log in self.detection_system.attack_log]
            detection_probs = [log['probability'] for log in self.detection_system.attack_log]
            
            plt.scatter(detection_times, detection_probs, c='red', alpha=0.6)
            plt.axhline(y=0.7, color='green', linestyle='--', label='Detection Threshold')
            plt.title('DoS Detection Timeline')
            plt.xlabel('Simulation Time (s)')
            plt.ylabel('Attack Probability')
            plt.legend()
        
        plt.tight_layout()
        plt.savefig('nano_network_results.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_detection_timeline(self):
        """Plot detailed detection timeline"""
        if not self.detection_system.attack_log:
            return
            
        plt.figure(figsize=(10, 6))
        
        detection_times = [log['timestamp'] for log in self.detection_system.attack_log]
        detection_probs = [log['probability'] for log in self.detection_system.attack_log]
        node_ids = [log['node_id'] for log in self.detection_system.attack_log]
        
        colors = ['red' if 'Malicious' in node_id else 'blue' for node_id in node_ids]
        
        plt.scatter(detection_times, detection_probs, c=colors, alpha=0.7, s=100)
        plt.axhline(y=0.7, color='green', linestyle='--', linewidth=2, label='Detection Threshold')
        
        # Add labels for malicious nodes
        for i, (time, prob, node_id) in enumerate(zip(detection_times, detection_probs, node_ids)):
            if 'Malicious' in node_id and prob > 0.7:
                plt.annotate(node_id, (time, prob), textcoords="offset points", 
                           xytext=(0,10), ha='center', fontsize=8)
        
        plt.title('AI/ML DoS Detection Results Over Time')
        plt.xlabel('Simulation Time (seconds)')
        plt.ylabel('Attack Probability')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig('dos_detection_timeline.png', dpi=300, bbox_inches='tight')
        plt.show()

def main():
    """Run the complete simulation"""
    # Set random seeds for reproducibility
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    
    # Create and run simulation
    simulation = NanoNetworkSimulation(
        num_normal_nodes=15,
        num_malicious_nodes=3, 
        simulation_time=60
    )
    
    simulation.run_simulation()
    
    # Save simulation data
    with open('simulation_data.json', 'w') as f:
        simulation_data = {
            'timestamp': datetime.now().isoformat(),
            'nodes': len(simulation.nodes),
            'malicious_nodes': len([n for n in simulation.nodes if n.is_malicious]),
            'attacks_detected': len(simulation.detection_system.attack_log),
            'mitigation_actions': len(simulation.detection_system.mitigation_actions)
        }
        json.dump(simulation_data, f, indent=2)

if __name__ == "__main__":
    main()