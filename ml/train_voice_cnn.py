# ml/train_voice_cnn.py
# Trains CNN model for voice authentication using MFCC features

import sys
import os

# Add backend directory to Python path
backend_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'backend')
sys.path.insert(0, backend_path)

import numpy as np
import pickle
from sqlalchemy.orm import Session

# Check if we're using TensorFlow or PyTorch
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, models
    USE_TENSORFLOW = True
    print("Using TensorFlow for CNN")
except ImportError:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    USE_TENSORFLOW = False
    print("Using PyTorch for CNN")

from database.db import SessionLocal
from database.models import User, VoiceTemplate


# ============================================================================
# PyTorch CNN Model (if TensorFlow not available)
# ============================================================================

class VoiceCNN_PyTorch(nn.Module):
    """Simple CNN for voice authentication using PyTorch"""
    
    def __init__(self):
        super(VoiceCNN_PyTorch, self).__init__()
        
        # Input: 13 MFCC features (single vector)
        self.fc1 = nn.Linear(13, 64)
        self.dropout1 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(64, 32)
        self.dropout2 = nn.Dropout(0.3)
        self.fc3 = nn.Linear(32, 16)
        self.fc4 = nn.Linear(16, 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout1(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout2(x)
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)
        x = self.sigmoid(x)
        return x


# ============================================================================
# TensorFlow CNN Model
# ============================================================================

def create_voice_cnn_tensorflow():
    """Simple CNN for voice authentication using TensorFlow"""
    
    model = models.Sequential([
        # Input: 13 MFCC features
        layers.Dense(64, activation='relu', input_shape=(13,)),
        layers.Dropout(0.3),
        
        layers.Dense(32, activation='relu'),
        layers.Dropout(0.3),
        
        layers.Dense(16, activation='relu'),
        
        layers.Dense(1, activation='sigmoid')  # Binary: genuine (1) or impostor (0)
    ])
    
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    return model


# ============================================================================
# Data Generation
# ============================================================================

def generate_training_data(genuine_mfcc, num_genuine=20, num_impostor=40):
    """Generate synthetic training data from enrolled MFCC"""
    
    genuine_mfcc = np.array(genuine_mfcc)
    
    # Generate genuine samples with small variations (±10%)
    genuine_samples = []
    for _ in range(num_genuine):
        noise = np.random.normal(1.0, 0.10, size=genuine_mfcc.shape)
        noisy_sample = genuine_mfcc * noise
        genuine_samples.append(noisy_sample)
    
    # Generate impostor samples with larger variations (±40%)
    impostor_samples = []
    for _ in range(num_impostor):
        noise = np.random.normal(1.0, 0.40, size=genuine_mfcc.shape)
        impostor_sample = genuine_mfcc * noise
        # Add random shift to make it more different
        shift = np.random.uniform(-50, 50, size=genuine_mfcc.shape)
        impostor_sample += shift
        impostor_samples.append(impostor_sample)
    
    # Combine and create labels
    X = np.array(genuine_samples + impostor_samples)
    y = np.array([1] * num_genuine + [0] * num_impostor)
    
    # Shuffle
    indices = np.random.permutation(len(X))
    X = X[indices]
    y = y[indices]
    
    return X, y


# ============================================================================
# Training Function - TensorFlow
# ============================================================================

def train_model_tensorflow(username: str):
    """Train voice CNN using TensorFlow"""
    
    db = SessionLocal()
    
    try:
        user = db.query(User).filter(User.username == username).first()
        if not user:
            print(f"User '{username}' not found!")
            return None
        
        template = db.query(VoiceTemplate).filter(
            VoiceTemplate.user_id == user.id
        ).first()
        
        if not template:
            print(f"No voice template for '{username}'!")
            return None
        
        print(f"\n{'='*60}")
        print(f"Training Voice CNN for: {username}")
        print(f"{'='*60}")
        print(f"Enrolled MFCC features: {len(template.mfcc_features)}")
        
        # Generate training data
        X_train, y_train = generate_training_data(template.mfcc_features)
        
        print(f"\nTraining data generated:")
        print(f"  Genuine samples: {np.sum(y_train == 1)}")
        print(f"  Impostor samples: {np.sum(y_train == 0)}")
        print(f"  Total samples: {len(X_train)}")
        
        # Create and train model
        print("\nBuilding CNN model...")
        model = create_voice_cnn_tensorflow()
        
        print(model.summary())
        
        print("\nTraining...")
        history = model.fit(
            X_train, y_train,
            epochs=50,
            batch_size=8,
            validation_split=0.2,
            verbose=1
        )
        
        # Evaluate
        final_loss = history.history['loss'][-1]
        final_acc = history.history['accuracy'][-1]
        val_loss = history.history['val_loss'][-1]
        val_acc = history.history['val_accuracy'][-1]
        
        print(f"\n{'='*60}")
        print(f"Training Complete!")
        print(f"{'='*60}")
        print(f"Final Training Accuracy: {final_acc:.2%}")
        print(f"Final Validation Accuracy: {val_acc:.2%}")
        print(f"Final Training Loss: {final_loss:.4f}")
        print(f"Final Validation Loss: {val_loss:.4f}")
        
        # Save model
        model_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), 'models'
        )
        os.makedirs(model_dir, exist_ok=True)
        
        model_path = os.path.join(model_dir, f"{username}_voice_cnn.h5")
        model.save(model_path)
        
        # Also save metadata
        metadata = {
            'username': username,
            'enrolled_mfcc': template.mfcc_features,
            'accuracy': float(val_acc),
            'loss': float(val_loss),
            'framework': 'tensorflow'
        }
        
        metadata_path = os.path.join(model_dir, f"{username}_voice_metadata.pkl")
        with open(metadata_path, 'wb') as f:
            pickle.dump(metadata, f)
        
        print(f"\n✅ Model saved to: {model_path}")
        print(f"✅ Metadata saved to: {metadata_path}")
        
        return model_path
        
    finally:
        db.close()


# ============================================================================
# Training Function - PyTorch
# ============================================================================

def train_model_pytorch(username: str):
    """Train voice CNN using PyTorch"""
    
    db = SessionLocal()
    
    try:
        user = db.query(User).filter(User.username == username).first()
        if not user:
            print(f"User '{username}' not found!")
            return None
        
        template = db.query(VoiceTemplate).filter(
            VoiceTemplate.user_id == user.id
        ).first()
        
        if not template:
            print(f"No voice template for '{username}'!")
            return None
        
        print(f"\n{'='*60}")
        print(f"Training Voice CNN for: {username}")
        print(f"{'='*60}")
        print(f"Enrolled MFCC features: {len(template.mfcc_features)}")
        
        # Generate training data
        X_train, y_train = generate_training_data(template.mfcc_features)
        
        print(f"\nTraining data generated:")
        print(f"  Genuine samples: {np.sum(y_train == 1)}")
        print(f"  Impostor samples: {np.sum(y_train == 0)}")
        print(f"  Total samples: {len(X_train)}")
        
        # Convert to PyTorch tensors
        X_tensor = torch.FloatTensor(X_train)
        y_tensor = torch.FloatTensor(y_train).unsqueeze(1)
        
        # Create model
        model = VoiceCNN_PyTorch()
        criterion = nn.BCELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        
        print("\nTraining...")
        epochs = 50
        batch_size = 8
        
        for epoch in range(epochs):
            model.train()
            total_loss = 0
            correct = 0
            total = 0
            
            # Mini-batch training
            indices = torch.randperm(len(X_tensor))
            for i in range(0, len(X_tensor), batch_size):
                batch_indices = indices[i:i+batch_size]
                batch_X = X_tensor[batch_indices]
                batch_y = y_tensor[batch_indices]
                
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                predicted = (outputs > 0.5).float()
                correct += (predicted == batch_y).sum().item()
                total += batch_y.size(0)
            
            avg_loss = total_loss / (len(X_tensor) / batch_size)
            accuracy = correct / total
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{epochs}] Loss: {avg_loss:.4f} Accuracy: {accuracy:.2%}")
        
        print(f"\n{'='*60}")
        print(f"Training Complete!")
        print(f"{'='*60}")
        print(f"Final Accuracy: {accuracy:.2%}")
        print(f"Final Loss: {avg_loss:.4f}")
        
        # Save model
        model_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), 'models'
        )
        os.makedirs(model_dir, exist_ok=True)
        
        model_path = os.path.join(model_dir, f"{username}_voice_cnn.pth")
        torch.save(model.state_dict(), model_path)
        
        # Save metadata
        metadata = {
            'username': username,
            'enrolled_mfcc': template.mfcc_features,
            'accuracy': float(accuracy),
            'loss': float(avg_loss),
            'framework': 'pytorch'
        }
        
        metadata_path = os.path.join(model_dir, f"{username}_voice_metadata.pkl")
        with open(metadata_path, 'wb') as f:
            pickle.dump(metadata, f)
        
        print(f"\n✅ Model saved to: {model_path}")
        print(f"✅ Metadata saved to: {metadata_path}")
        
        return model_path
        
    finally:
        db.close()


# ============================================================================
# Main Entry Point
# ============================================================================

def train_voice_model(username: str):
    """Train voice CNN model for a user"""
    if USE_TENSORFLOW:
        return train_model_tensorflow(username)
    else:
        return train_model_pytorch(username)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        username = sys.argv[1]
    else:
        username = input("Enter username to train voice model for: ").strip()
    
    train_voice_model(username)