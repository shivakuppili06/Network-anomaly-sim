import pandas as pd
import numpy as np
import os

def generate_ar1_series(periods, mean, std, rho=0.8):
    """Generate an AR(1) time series."""
    # Variance of the process is sigma_eps^2 / (1 - rho^2)
    # We want variance to be std^2, so sigma_eps = std * sqrt(1 - rho^2)
    sigma_eps = std * np.sqrt(1 - rho**2)
    eps = np.random.normal(0, sigma_eps, periods)
    series = np.zeros(periods)
    series[0] = np.random.normal(0, std)
    for t in range(1, periods):
        series[t] = rho * series[t-1] + eps[t]
    return series + mean

def generate_network_data(days=30, interval_min=1):
    print("Generating conceptual synthetic network metrics with realistic noise...")
    
    periods = days * 24 * 60 // interval_min
    start_date = pd.Timestamp('2023-01-01')
    timestamps = pd.date_range(start=start_date, periods=periods, freq=f'{interval_min}min')
    
    # Base metrics with autocorrelation to prevent trivial separability
    latency = generate_ar1_series(periods, mean=20, std=4, rho=0.85)
    
    # Packet loss: heavily skewed, lots of zeros, some low values.
    packet_loss_base = generate_ar1_series(periods, mean=-0.2, std=0.6, rho=0.7)
    packet_loss = np.clip(packet_loss_base, 0, None) 
    
    throughput = generate_ar1_series(periods, mean=1000, std=80, rho=0.9)
    jitter = generate_ar1_series(periods, mean=5, std=2, rho=0.8)
    
    df = pd.DataFrame({
        'timestamp': timestamps,
        'latency_ms': latency,
        'packet_loss_pct': packet_loss,
        'throughput_mbps': throughput,
        'jitter_ms': jitter,
        'is_anomaly': 0
    })
    
    # Inject anomalies (~2% of data points affected)
    # We aim for around 2% of the total periods to be marked as anomalous.
    # We create events that span multiple periods for drift and shift.
    target_anomalous_points = int(periods * 0.02)
    avg_event_length = 8
    num_events = max(10, target_anomalous_points // avg_event_length)
    
    event_start_indices = np.random.choice(periods - 30, num_events, replace=False)
    
    for start_idx in event_start_indices:
        anomaly_category = np.random.choice(['spike', 'drift', 'shift'])
        metric = np.random.choice(['latency_ms', 'throughput_mbps', 'jitter_ms', 'packet_loss_pct'])
        
        if anomaly_category == 'spike':
            length = np.random.randint(1, 4)
            for i in range(length):
                idx = start_idx + i
                if metric == 'latency_ms':
                    df.loc[idx, metric] += np.random.uniform(10, 30)
                elif metric == 'throughput_mbps':
                    df.loc[idx, metric] *= np.random.uniform(0.7, 0.85)
                elif metric == 'jitter_ms':
                    df.loc[idx, metric] += np.random.uniform(5, 12)
                elif metric == 'packet_loss_pct':
                    df.loc[idx, metric] += np.random.uniform(1, 4)
                df.loc[idx, 'is_anomaly'] = 1
                
        elif anomaly_category == 'drift':
            length = np.random.randint(10, 20)
            drift_factor = np.linspace(0, 1, length)
            for i in range(length):
                idx = start_idx + i
                df.loc[idx, 'is_anomaly'] = 1
                if metric == 'latency_ms':
                    df.loc[idx, metric] += drift_factor[i] * np.random.uniform(15, 30)
                elif metric == 'throughput_mbps':
                    df.loc[idx, metric] -= drift_factor[i] * np.random.uniform(150, 300)
                elif metric == 'jitter_ms':
                    df.loc[idx, metric] += drift_factor[i] * np.random.uniform(6, 12)
                elif metric == 'packet_loss_pct':
                    df.loc[idx, metric] += drift_factor[i] * np.random.uniform(1.5, 4.5)
                    
        elif anomaly_category == 'shift':
            length = np.random.randint(10, 20)
            for i in range(length):
                idx = start_idx + i
                df.loc[idx, 'is_anomaly'] = 1
                if metric == 'latency_ms':
                    df.loc[idx, metric] += np.random.uniform(10, 25)
                elif metric == 'throughput_mbps':
                    df.loc[idx, metric] -= np.random.uniform(150, 250)
                elif metric == 'jitter_ms':
                    df.loc[idx, metric] += np.random.uniform(5, 10)
                elif metric == 'packet_loss_pct':
                    df.loc[idx, metric] += np.random.uniform(1.5, 3)

    # Final clip
    df['latency_ms'] = np.clip(df['latency_ms'], 1, None)
    df['packet_loss_pct'] = np.clip(df['packet_loss_pct'], 0, 100)
    df['throughput_mbps'] = np.clip(df['throughput_mbps'], 10, None)
    df['jitter_ms'] = np.clip(df['jitter_ms'], 0, None)
        
    os.makedirs('data', exist_ok=True)
    output_path = 'data/network_metrics.csv'
    df.to_csv(output_path, index=False)
    print(f"Data saved to {output_path}")

if __name__ == '__main__':
    generate_network_data()
