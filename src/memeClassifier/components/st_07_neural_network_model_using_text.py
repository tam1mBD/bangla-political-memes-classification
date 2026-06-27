import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from memeClassifier import logger
from memeClassifier.entity.config_entity import NeuralNetworkModelUsingTextConfig


class SimpleNN(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(SimpleNN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        return self.network(x)

class NeuralNetworkModelUsingText:
    def __init__(self, config: NeuralNetworkModelUsingTextConfig):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def initiate_model_training(self):
        logger.info("Loading feature datasets")
        train_df = pd.read_csv(self.config.train_features_csv)
        
        X = train_df[['political_specific_count', 'political_specific_ratio']].to_numpy(dtype=np.float32)
        y = train_df['Label'].map({'Political': 1, 'NonPolitical': 0}).to_numpy(dtype=np.int64)
        
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        logger.info(f"Training set size: {X_train.shape[0]}")
        logger.info(f"Validation set size: {X_val.shape[0]}")
        
        # Convert to PyTorch tensors
        X_train_tensor = torch.FloatTensor(X_train).to(self.device)
        y_train_tensor = torch.FloatTensor(y_train).unsqueeze(1).to(self.device)
        X_val_tensor = torch.FloatTensor(X_val).to(self.device)
        y_val_tensor = torch.FloatTensor(y_val).unsqueeze(1).to(self.device)
        
        # Create DataLoader
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=self.config.batch_size, shuffle=True)
        
        # Initialize model, loss function, optimizer
        model = SimpleNN(input_dim=X_train.shape[1], hidden_dim=self.config.hidden_dim).to(self.device)
        criterion = nn.BCELoss()
        optimizer = optim.Adam(model.parameters(), lr=self.config.learning_rate)
        
        logger.info(f"Training Neural Network for {self.config.epochs} epochs...")
        for epoch in range(self.config.epochs):
            model.train()
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
        
        # Evaluation
        model.eval()
        with torch.no_grad():
            train_outputs = model(X_train_tensor)
            train_preds = (train_outputs >= 0.5).float().cpu().numpy()
            
            val_outputs = model(X_val_tensor)
            val_preds = (val_outputs >= 0.5).float().cpu().numpy()
        
        train_accuracy = accuracy_score(y_train, train_preds)
        val_accuracy = accuracy_score(y_val, val_preds)
        
        logger.info(f"Training Accuracy: {train_accuracy:.4f}")
        logger.info(f"Validation Accuracy: {val_accuracy:.4f}")
        logger.info(f"Classification Report (Validation Set):\n{classification_report(y_val, val_preds, target_names=['NonPolitical', 'Political'])}")
        
        torch.save(model.state_dict(), self.config.model_path)
        logger.info(f"PyTorch Neural Network Model saved to {self.config.model_path}")