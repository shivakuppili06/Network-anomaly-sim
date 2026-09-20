import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import json
import os
import pickle

# Note: This is a conceptual demonstration.

class LSTMAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim=16):
        super(LSTMAutoencoder, self).__init__()
        self.encoder = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.decoder = nn.LSTM(hidden_dim, input_dim, batch_first=True)
        
    def forward(self, x):
        _, (hidden, _) = self.encoder(x)
        repeated_hidden = hidden[-1].unsqueeze(1).repeat(1, x.size(1), 1)
        out, _ = self.decoder(repeated_hidden)
        return out

def evaluate_model(y_true, y_pred, y_scores):
    return {
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0),
        'roc_auc': roc_auc_score(y_true, y_scores)
    }

def train_models():
    print("Training models on synthetic data...")
    df = pd.read_csv('data/network_metrics.csv')
    
    features = ['latency_ms', 'packet_loss_pct', 'throughput_mbps', 'jitter_ms']
    X = df[features].values
    y_true = df['is_anomaly'].values
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    os.makedirs('models', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    
    # Isolation Forest
    print("Training Isolation Forest...")
    iso_forest = IsolationForest(contamination=0.02, random_state=42)
    iso_forest.fit(X_scaled)
    
    scores_if = -iso_forest.score_samples(X_scaled)
    preds_if = (iso_forest.predict(X_scaled) == -1).astype(int)
    
    with open('models/iso_forest.pkl', 'wb') as f:
        pickle.dump(iso_forest, f)
        
    metrics_if = evaluate_model(y_true, preds_if, scores_if)
    
    # LSTM Autoencoder
    print("Training LSTM Autoencoder...")
    # Very simple sequence creation for demonstration
    seq_length = 5
    X_seq = []
    y_seq_true = []
    for i in range(len(X_scaled) - seq_length):
        X_seq.append(X_scaled[i:i+seq_length])
        y_seq_true.append(y_true[i+seq_length-1])
        
    X_seq = np.array(X_seq)
    y_seq_true = np.array(y_seq_true)
    
    tensor_X = torch.FloatTensor(X_seq)
    dataset = TensorDataset(tensor_X, tensor_X)
    loader = DataLoader(dataset, batch_size=64, shuffle=True)
    
    model = LSTMAutoencoder(input_dim=len(features))
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    epochs = 3
    for epoch in range(epochs):
        for batch_x, _ in loader:
            optimizer.zero_grad()
            output = model(batch_x)
            loss = criterion(output, batch_x)
            loss.backward()
            optimizer.step()
            
    torch.save(model.state_dict(), 'models/lstm_autoencoder.pth')
    
    # Evaluate LSTM
    model.eval()
    with torch.no_grad():
        reconstructed = model(tensor_X)
        mse = torch.mean(torch.pow(tensor_X - reconstructed, 2), dim=2)
        scores_lstm = mse[:, -1].numpy()
        
    threshold = np.percentile(scores_lstm, 98)
    preds_lstm = (scores_lstm > threshold).astype(int)
    
    metrics_lstm = evaluate_model(y_seq_true, preds_lstm, scores_lstm)
    
    report = {
        'isolation_forest': metrics_if,
        'lstm_autoencoder': metrics_lstm
    }
    
    with open('results/detection_report.json', 'w') as f:
        json.dump(report, f, indent=4)
        
    print("Models trained and evaluated. Results saved to results/detection_report.json")

if __name__ == '__main__':
    train_models()
