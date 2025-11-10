import torch
import torch.nn as nn
import numpy as np
from collections import deque
import json
import time

class NanoDoSDetector(nn.Module):
    def __init__(self, input_features=5):
        super(NanoDoSDetector, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_features, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(), 
            nn.Linear(8, 2),
            nn.Softmax(dim=1)
        )
        
    def forward(self, x):
        return self.network(x)

class RealTimeDetector:
    def __init__(self):
        self.model = NanoDoSDetector()
        try:
            self.model.load_state_dict(torch.load('nano_dos_model.pth'))
            print("AI Model loaded successfully")
        except:
            print("Initializing new model")
        
        self.model.eval()
        self.feature_window = deque(maxlen=10)
        self.attack_threshold = 0.75
        
        # Feature normalization parameters
        self.feature_mean = np.array([1000, 500, 0.05, 50, 0.1])  # Example values
        self.feature_std = np.array([500, 300, 0.1, 30, 0.2])
    
    def extract_features(self, packet_rate, avg_packet_size, error_rate, 
                        connection_attempts, response_time):
        """Extract and normalize features for ML model"""
        features = np.array([
            packet_rate,
            avg_packet_size, 
            error_rate,
            connection_attempts,
            response_time
        ])
        
        # Normalize features
        normalized_features = (features - self.feature_mean) / self.feature_std
        return normalized_features.astype(np.float32)
    
    def detect_dos_attack(self, packet_rate, avg_packet_size, error_rate):
        """Main detection function called from NS-3"""
        # Simulate additional features for demo
        connection_attempts = min(packet_rate / 10, 100)  # Derived feature
        response_time = max(0.001, 1.0 / packet_rate) if packet_rate > 0 else 1.0
        
        features = self.extract_features(
            packet_rate, avg_packet_size, error_rate,
            connection_attempts, response_time
        )
        
        with torch.no_grad():
            input_tensor = torch.FloatTensor(features).unsqueeze(0)
            output = self.model(input_tensor)
            attack_prob = output[0][1].item()  # Probability of attack class
            
            # Log detection for analysis
            self.log_detection(packet_rate, attack_prob)
            
            return attack_prob > self.attack_threshold
    
    def log_detection(self, packet_rate, attack_prob):
        """Log detection results for analysis"""
        log_entry = {
            'timestamp': time.time(),
            'packet_rate': packet_rate,
            'attack_probability': attack_prob,
            'is_attack': attack_prob > self.attack_threshold
        }
        
        with open('detection_log.json', 'a') as f:
            f.write(json.dumps(log_entry) + '\n')

# Global detector instance
detector = RealTimeDetector()

def detect_dos_attack(packet_rate, avg_packet_size, error_rate):
    """Interface function for NS-3 calls"""
    return detector.detect_dos_attack(packet_rate, avg_packet_size, error_rate)

if __name__ == "__main__":
    # Test the detector
    print("Testing DoS detector with sample data...")
    test_result = detect_dos_attack(5000, 1000, 0.01)  # High rate = likely attack
    print(f"Detection result: {test_result}")