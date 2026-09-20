import pandas as pd
import matplotlib.pyplot as plt
import json
import os

def generate_visualizations():
    print("Generating visualizations...")
    os.makedirs('results', exist_ok=True)
    
    df = pd.read_csv('data/network_metrics.csv')
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Plot 1: Time series with anomalies (just first 1000 points for clarity)
    plot_df = df.head(1000)
    plt.figure(figsize=(15, 6))
    plt.plot(plot_df['timestamp'], plot_df['latency_ms'], label='Latency (ms)', alpha=0.7)
    
    anomalies = plot_df[plot_df['is_anomaly'] == 1]
    plt.scatter(anomalies['timestamp'], anomalies['latency_ms'], color='red', label='Anomaly', zorder=5)
    
    plt.title('Synthetic Network Latency with Injected Anomalies')
    plt.xlabel('Time')
    plt.ylabel('Latency (ms)')
    plt.legend()
    plt.tight_layout()
    plt.savefig('results/timeseries_anomalies.png')
    
    # Plot 2: Metrics Comparison
    if os.path.exists('results/detection_report.json'):
        with open('results/detection_report.json', 'r') as f:
            report = json.load(f)
            
        metrics = ['precision', 'recall', 'f1', 'roc_auc']
        if_scores = [report['isolation_forest'][m] for m in metrics]
        lstm_scores = [report['lstm_autoencoder'][m] for m in metrics]
        
        x = range(len(metrics))
        width = 0.35
        
        plt.figure(figsize=(10, 6))
        plt.bar([i - width/2 for i in x], if_scores, width, label='Isolation Forest')
        plt.bar([i + width/2 for i in x], lstm_scores, width, label='LSTM Autoencoder')
        
        plt.xlabel('Metrics')
        plt.ylabel('Score')
        plt.title('Model Comparison')
        plt.xticks(x, metrics)
        plt.legend()
        plt.tight_layout()
        plt.savefig('results/model_comparison.png')
        
    print("Visualizations saved to results/")

if __name__ == '__main__':
    generate_visualizations()
