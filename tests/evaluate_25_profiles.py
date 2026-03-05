import json
from agents.pipeline import MultiAgentPipeline
from agents.schema import QuoteInput

def evaluate():
    pipeline = MultiAgentPipeline(models_dir="models")
    with open("tests/synthetic_data.json", "r") as f:
        profiles = json.load(f)
        
    print(f"\n🔍 Evaluating {len(profiles)} Synthetic Edge-case Profiles through the 4-Agent Pipeline...\n")
    
    correct = 0
    
    for i, profile in enumerate(profiles):
        print(f"--- Profile {i+1}: {profile['Profile_Name']} ---")
        quote = QuoteInput(**profile)
        result = pipeline.execute(quote)
        
        tier = result.risk_evaluation.risk_tier
        conv = result.conversion_prediction.conversion_band
        prem = result.premium_advice.premium_issue
        decision = result.final_decision.decision
        esc = result.escalation_status.escalation_required
        
        print(f"  Risk Tier: {tier} | Conversion: {conv} | Premium Issue: {prem}")
        print(f"  Final Decision: {decision} | Escalate: {esc}")
        
        # Verify Accuracy against known profile category constraints
        is_accurate = False
        if "Low_Risk" in profile['Profile_Name']:
            # Low risk profiles should never trigger escalation unless confidence is awful
            if esc == False and tier in ["LOW", "MEDIUM"]:
                is_accurate = True
        elif "High_Risk" in profile['Profile_Name']:
            # High risk profiles MUST be escalated
            if tier == "HIGH" and decision == "ESCALATE_TO_UNDERWRITER":
                is_accurate = True
        elif "Medium_Risk" in profile['Profile_Name']:
            if tier in ["MEDIUM", "HIGH"]: 
                is_accurate = True
        elif "Premium_Sensitive" in profile['Profile_Name']:
            # Must catch the premium discrepancy
            if prem == True:
                is_accurate = True
        elif "Escalation_Guarantee" in profile['Profile_Name']:
            # Must strictly enforce human escalation protocol
            if esc == True:
                is_accurate = True
                
        if is_accurate:
            correct += 1
            print("  ✅ VERIFIED ACCURATE\n")
        else:
            print("  ❌ LOGIC MISMATCH\n")
            
    print("==================================================")
    print(f"🎯 Total Architecture Accuracy: {correct}/{len(profiles)} ({(correct/len(profiles))*100}%)")
    print("==================================================")

if __name__ == "__main__":
    evaluate()
