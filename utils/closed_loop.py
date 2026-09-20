import pandas as pd
import numpy as np

# Rule-based simulation of closed-loop automation
# Not a live control system

def simulate_closed_loop():
    print("Simulating closed-loop automation...")
    df = pd.read_csv('data/network_metrics.csv')
    
    # Using ground truth for demonstration of rules, 
    # in reality this would be model predictions.
    anomalies = df[df['is_anomaly'] == 1].copy()
    
    events = []
    
    for _, row in anomalies.iterrows():
        action = "Log only"
        
        # Simple heuristic mapping
        if row['latency_ms'] > 50 and row['packet_loss_pct'] > 5:
            action = "reroute traffic via secondary path"
        elif row['throughput_mbps'] < 500:
            action = "flag node for restart"
        elif row['jitter_ms'] > 20:
            action = "adjust QoS buffering"
            
        events.append(f"{row['timestamp']} - Anomaly Detected -> ACTION: {action}")
        
    with open('results/closed_loop_events.log', 'w') as f:
        f.write("\n".join(events))
        
    print(f"Generated {len(events)} events in results/closed_loop_events.log")

if __name__ == '__main__':
    simulate_closed_loop()
