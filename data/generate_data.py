import pandas as pd
import numpy as np
import os

def generate_network_data(days=30, interval_min=1):
    print("Generating conceptual synthetic network metrics...")
    
    # Time index
    periods = days * 24 * 60 // interval_min
    start_date = pd.Timestamp('2023-01-01')
    timestamps = pd.date_range(start=start_date, periods=periods, freq=f'{interval_min}min')
    
    # Base metrics
    latency = np.random.normal(20, 5, periods) # mean 20ms, std 5ms
    packet_loss = np.random.exponential(0.1, periods) # low packet loss
    throughput = np.random.normal(1000, 100, periods) # 1000 Mbps
    jitter = np.random.normal(5, 2, periods)
    
    # Clip to realistic bounds
    latency = np.clip(latency, 1, None)
    packet_loss = np.clip(packet_loss, 0, 100)
    throughput = np.clip(throughput, 10, None)
    jitter = np.clip(jitter, 0, None)
    
    df = pd.DataFrame({
        'timestamp': timestamps,
        'latency_ms': latency,
        'packet_loss_pct': packet_loss,
        'throughput_mbps': throughput,
        'jitter_ms': jitter,
        'is_anomaly': 0
    })
    
    # Inject anomalies (~2% of data points)
    num_anomalies = int(periods * 0.02)
    anomaly_indices = np.random.choice(periods, num_anomalies, replace=False)
    
    for idx in anomaly_indices:
        anomaly_type = np.random.choice(['latency_spike', 'throughput_drop', 'jitter_spike', 'packet_loss_spike'])
        
        if anomaly_type == 'latency_spike':
            df.loc[idx, 'latency_ms'] += np.random.uniform(50, 150)
        elif anomaly_type == 'throughput_drop':
            df.loc[idx, 'throughput_mbps'] *= np.random.uniform(0.1, 0.4)
        elif anomaly_type == 'jitter_spike':
            df.loc[idx, 'jitter_ms'] += np.random.uniform(20, 50)
        elif anomaly_type == 'packet_loss_spike':
            df.loc[idx, 'packet_loss_pct'] += np.random.uniform(5, 20)
            
        df.loc[idx, 'is_anomaly'] = 1
        
    # Ensure directory exists
    os.makedirs('data', exist_ok=True)
    
    # Save
    output_path = 'data/network_metrics.csv'
    df.to_csv(output_path, index=False)
    print(f"Data saved to {output_path}")

if __name__ == '__main__':
    generate_network_data()
