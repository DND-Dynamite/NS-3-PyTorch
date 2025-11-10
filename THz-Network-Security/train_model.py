# train_model.py
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from nano_network_simulator import DoSDetector
import json

def generate_training_data(num_samples=10000):
    """Generate synthetic training data for DoS detection"""
    normal_traffic = np.random.exponential(1.0, (num_samples//2, 8))
    attack_traffic = np.random.exponential(5.0, (num_samples//2, 8))
    
    # Normal traffic features (lower values)
    normal_traffic[:, 0] = np.random.normal(10, 5, num_samples//2)  # packet rate
    normal_traffic[:, 1] = np.random.normal(500, 100, num_samples//2)  # packet size
    
    # Attack traffic features (higher values)
    attack_traffic[:, 0] = np.random.normal(100, 30, num_samples//2)  # high packet rate
    attack_traffic[:, 1] = np.random.normal(1500, 500, num_samples//2)  # large packets
    
    X = np.vstack([normal_traffic, attack_traffic])
    y = np.hstack([np.zeros(num_samples//2), np.ones(num_samples//2)])
    
    return X, y

def train_model():
    """Train the DoS detection model"""
    print("Generating training data...")
    X, y = generate_training_data()
    
    # Convert to PyTorch tensors
    X_tensor = torch.FloatTensor(X)
    y_tensor = torch.LongTensor(y)
    
    # Create model and optimizer
    model = DoSDetector()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Training loop
    model.train()
    for epoch in range(100):
        optimizer.zero_grad()
        outputs = model(X_tensor)
        loss = criterion(outputs, y_tensor)
        loss.backward()
        optimizer.step()
        
        if epoch % 10 == 0:
            print(f'Epoch {epoch}, Loss: {loss.item():.4f}')
    
    # Save model
    torch.save(model.state_dict(), 'nano_dos_model.pth')
    print("Model trained and saved successfully!")
    
    # Test accuracy
    model.eval()
    with torch.no_grad():
        outputs = model(X_tensor)
        _, predicted = torch.max(outputs, 1)
        accuracy = (predicted == y_tensor).float().mean()
        print(f'Training accuracy: {accuracy.item()*100:.2f}%')

if __name__ == "__main__":
    train_model()