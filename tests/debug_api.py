import json
import requests
from agents.pipeline import MultiAgentPipeline
from agents.schema import QuoteInput

def debug():
    pipeline = MultiAgentPipeline(models_dir="models")
    with open("tests/synthetic_data.json", "r") as f:
        profiles = json.load(f)
        
    high_risk_1 = next(p for p in profiles if p["Profile_Name"] == "High_Risk_1")
    
    # 1. Local
    print("--- LOCAL PIPELINE EXECUTION ---")
    quote = QuoteInput(**high_risk_1)
    res_local = pipeline.execute(quote)
    print(f"Risk Tier: {res_local.risk_evaluation.risk_tier}")
    
    # 2. API 
    allowed_fields = list(quote.model_dump().keys())
    hr_payload = {k: v for k, v in high_risk_1.items() if k in allowed_fields}
    print("\n--- FASTAPI ENDPOINT EXECUTION ---")
    res_api = requests.post("http://127.0.0.1:8000/api/v1/evaluate_quote", json=hr_payload).json()
    print(f"Risk Tier: {res_api['risk_evaluation']['risk_tier']}")

if __name__ == "__main__":
    debug()
