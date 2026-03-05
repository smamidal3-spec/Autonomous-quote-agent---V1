import json
import pytest
import os
from agents.pipeline import MultiAgentPipeline
from agents.schema import QuoteInput

# Load pipeline once for testing
try:
    pipeline = MultiAgentPipeline(models_dir="models")
except Exception as e:
    pipeline = None
    
@pytest.fixture(scope="module")
def synthetic_data():
    with open("tests/synthetic_data.json", "r") as f:
        return json.load(f)

def test_pipeline_loaded():
    assert pipeline is not None, "Pipeline failed to initialize (models missing?)"

def test_high_risk_never_defaults_to_low(synthetic_data):
    """
    Ensure High Risk profiles (accidents, citations, young) do not slip through as low risk.
    """
    for profile in synthetic_data:
        if "High_Risk" in profile["Profile_Name"]:
            quote = QuoteInput(**profile)
            result = pipeline.execute(quote)
            
            # The risk tier should be HIGH, or at worst MEDIUM, never LOW
            assert result.risk_evaluation.risk_tier in ["HIGH", "MEDIUM"], \
                f"Failed Profile {profile['Profile_Name']}: Assigned {result.risk_evaluation.risk_tier} instead of HIGH/MEDIUM"
                
            # If HIGH, router should Escalate!
            if result.risk_evaluation.risk_tier == "HIGH":
                assert result.final_decision.decision == "ESCALATE_TO_UNDERWRITER" or result.escalation_status.escalation_required

def test_premium_sensitive_advises_discount(synthetic_data):
    """
    Ensure that low salary with high premium successfully triggers the Premium Advisor agent.
    """
    for profile in synthetic_data:
        if "Premium_Sensitive" in profile["Profile_Name"]:
            quote = QuoteInput(**profile)
            result = pipeline.execute(quote)
            
            # Agent 3 should flag a premium issue
            assert result.premium_advice.premium_issue == True, \
                f"{profile['Profile_Name']} should have flagged a premium issue due to salary."

def test_auto_approve_logic(synthetic_data):
    """
    Ensure Low Risk / High Conversion successfully auto-approves.
    """
    for profile in synthetic_data:
        if "Low_Risk" in profile["Profile_Name"]:
            quote = QuoteInput(**profile)
            result = pipeline.execute(quote)
            
            # If Risk is LOW and Conversion is HIGH, must auto approve
            if result.risk_evaluation.risk_tier == "LOW" and result.conversion_prediction.conversion_band == "HIGH":
                assert result.final_decision.decision == "AUTO_APPROVE"
                
def test_escalation_protocol(synthetic_data):
    """
    Ensure the escalation boolean is True whenever we hit the guarantee case.
    """
    for profile in synthetic_data:
        if "Escalation_Guarantee" in profile["Profile_Name"]:
            quote = QuoteInput(**profile)
            result = pipeline.execute(quote)
            
            assert result.escalation_status.escalation_required == True, \
                f"{profile['Profile_Name']} failed to trigger human escalation."
