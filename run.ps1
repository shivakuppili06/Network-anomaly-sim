Write-Host "Starting Network Anomaly Simulation Pipeline..."
Write-Host "------------------------------------------------"

Write-Host "1. Generating synthetic data..."
python data/generate_data.py

Write-Host "`n2. Training models..."
python models/train_detector.py

Write-Host "`n3. Simulating closed-loop automation..."
python utils/closed_loop.py

Write-Host "`n4. Simulating adaptive response (Contextual Bandit)..."
python utils/adaptive_response.py

Write-Host "`n5. Generating visualizations..."
python utils/visualize.py

Write-Host "`n------------------------------------------------"
Write-Host "Pipeline execution complete! Check the 'results/' directory for outputs."
