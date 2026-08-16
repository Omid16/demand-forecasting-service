from fastapi import FastAPI

app = FastAPI(title = "Demand Forecasting Service")

@app.get("/health")
def health():
    return {"status": "ok"}