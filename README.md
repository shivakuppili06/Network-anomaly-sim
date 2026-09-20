# Network Anomaly Simulation

This is a conceptual, simulated project demonstrating AI-driven anomaly detection and closed-loop automation concepts for network management.

**IMPORTANT:** This project uses entirely synthetic data for demonstration purposes. It does not contain or utilize real telecom or network data.

## Project Structure

- `data/generate_data.py`: Generates synthetic time-series network metrics (latency, packet loss, throughput, jitter) and injects anomalies.
- `models/train_detector.py`: Trains an unsupervised Isolation Forest (scikit-learn) and an LSTM Autoencoder (PyTorch) to detect anomalies.
- `utils/closed_loop.py`: A rule-based simulation demonstrating how detected anomalies could trigger corrective actions. This is NOT a live control system.
- `utils/visualize.py`: Generates plots comparing the models and visualizing the time series data.

## How to Run

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Generate Synthetic Data:**
   ```bash
   python data/generate_data.py
   ```
   This creates `data/network_metrics.csv`.

3. **Train Models and Evaluate:**
   ```bash
   python models/train_detector.py
   ```
   This trains the models and saves the evaluation report in `results/detection_report.json`.

4. **Simulate Closed-Loop Actions:**
   ```bash
   python utils/closed_loop.py
   ```
   This generates an event log in `results/closed_loop_events.log`.

5. **Generate Visualizations:**
   ```bash
   python utils/visualize.py
   ```
   This creates plots in the `results/` directory.

## Results Summary
The simulation outputs can be found in the `results/` directory, including model performance comparisons and simulated event logs mapping anomalies to actions.
