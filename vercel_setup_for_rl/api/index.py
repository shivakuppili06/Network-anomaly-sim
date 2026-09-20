from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Allow CORS so your frontend (index.html) can call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/api/health")
def health_check():
    return {"status": "ok", "message": "Python backend is running on Vercel!"}

@app.post("/api/train")
def trigger_training():
    # You can import and run functions from your train.py here
    # import train
    # result = train.run_training()
    return {"status": "success", "message": "Training endpoint hit"}
