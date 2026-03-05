import json
from agents.schema import QuoteInput

def debug():
    with open("tests/synthetic_data.json", "r") as f:
        profiles = json.load(f)
        
    high_risk_1 = next(p for p in profiles if p["Profile_Name"] == "High_Risk_1")
    quote = QuoteInput(**high_risk_1)
    
    print("Keys in profile JSON:", len(high_risk_1.keys()))
    print("Keys in model_dump:", len(quote.model_dump().keys()))

    print(quote.model_dump().keys())

if __name__ == "__main__":
    debug()
