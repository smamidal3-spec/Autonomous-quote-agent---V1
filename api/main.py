from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from agents.schema import QuoteInput, PipelineOutput
from agents.pipeline import MultiAgentPipeline

app = FastAPI(title="Autonomous Quote Agents API", version="1.0")

# Load pipeline once
try:
    pipeline = MultiAgentPipeline(models_dir="models")
except Exception as e:
    print(f"Failed to load pipeline: {e}")
    pipeline = None

@app.get("/")
def read_root():
    return {"status": "ok", "message": "Autonomous Quote Agents API is running."}

@app.post("/api/v1/evaluate_quote", response_model=PipelineOutput)
def evaluate_quote(quote: QuoteInput):
    if not pipeline:
        raise HTTPException(status_code=500, detail="Machine learning models not found or loaded.")
    
    try:
        # Run Quote through 4-Agent Pipeline
        result = pipeline.execute(quote)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
