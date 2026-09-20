from fastapi import FastAPI
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
import os
import json

app = FastAPI(title="Network Anomaly Sim API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def home():
    return {"message": "Welcome to the Network Anomaly Simulation API", "endpoints": ["/api/report", "/api/bandit", "/api/visualizations/timeseries"]}

@app.get("/api/report")
def get_detection_report():
    if os.path.exists("results/detection_report.json"):
        with open("results/detection_report.json", "r") as f:
            return json.load(f)
    return JSONResponse(status_code=404, content={"error": "Report not found"})

@app.get("/api/bandit")
def get_bandit_report():
    if os.path.exists("results/bandit_report.json"):
        with open("results/bandit_report.json", "r") as f:
            return json.load(f)
    return JSONResponse(status_code=404, content={"error": "Report not found"})

@app.get("/api/visualizations/timeseries")
def get_timeseries_image():
    if os.path.exists("results/timeseries_anomalies.png"):
        return FileResponse("results/timeseries_anomalies.png")
    return JSONResponse(status_code=404, content={"error": "Image not found"})

@app.get("/api/visualizations/comparison")
def get_comparison_image():
    if os.path.exists("results/model_comparison.png"):
        return FileResponse("results/model_comparison.png")
    return JSONResponse(status_code=404, content={"error": "Image not found"})

@app.get("/api/visualizations/bandit")
def get_bandit_image():
    if os.path.exists("results/bandit_performance.png"):
        return FileResponse("results/bandit_performance.png")
    return JSONResponse(status_code=404, content={"error": "Image not found"})
