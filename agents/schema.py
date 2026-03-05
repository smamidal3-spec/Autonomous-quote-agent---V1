from pydantic import BaseModel
from typing import List, Optional

# --- Shared Inputs ---
class QuoteInput(BaseModel):
    Agent_Type: str = "EA"
    Q_Creation_DT: str = "2019/10/01"
    Q_Valid_DT: str = "2023/12/31"
    Policy_Bind_DT: str = "2019/10/02"
    Region: str = "A"
    Agent_Num: int = 10
    Policy_Type: str = "Truck"
    HH_Vehicles: int = 1
    HH_Drivers: int = 1
    Driver_Age: float = 30.0
    Driving_Exp: int = 5
    Prev_Accidents: int = 0
    Prev_Citations: int = 0
    Gender: str = "Male"
    Marital_Status: str = "Married"
    Education: str = "Bachelors"
    Sal_Range: str = "50 K - 75 K"
    Coverage: str = "Balanced"
    Veh_Usage: str = "Business"
    Annual_Miles_Range: str = "<= 7.5 K"
    Vehicl_Cost_Range: str = "10 K - 20 K"
    Re_Quote: str = "No"
    Quoted_Premium: int = 1000

# --- Agent 1: Risk Profiler ---
class RiskOutput(BaseModel):
    risk_score: float
    risk_tier: str  # "LOW", "MEDIUM", "HIGH"
    risk_explanation: str

# --- Agent 2: Conversion Predictor ---
class ConversionOutput(BaseModel):
    conversion_probability: float  # 0-100
    conversion_band: str  # "LOW", "MEDIUM", "HIGH"
    top_conversion_drivers: List[str]

# --- Agent 3: Premium Advisor ---
class PremiumOutput(BaseModel):
    premium_issue: bool
    recommended_premium_range: List[float]  # [min, max]
    recommendation_reason: str

# --- Agent 4: Decision Router ---
class RoutingExplanation(BaseModel):
    risk_factor: str
    conversion_driver: str
    premium_issue: str

class DecisionOutput(BaseModel):
    decision: str  # "AUTO_APPROVE", "FOLLOW_UP_AGENT", "ESCALATE_TO_UNDERWRITER"
    confidence_score: float  # 0.0 - 1.0
    decision_explanation: str
    detailed_explanation: Optional[RoutingExplanation] = None

# --- Human Escalation Standard ---
class EscalationOutput(BaseModel):
    escalation_required: bool
    reason: str

# --- Full Pipeline Output ---
class PipelineOutput(BaseModel):
    quote_id: str
    risk_evaluation: RiskOutput
    conversion_prediction: ConversionOutput
    premium_advice: PremiumOutput
    final_decision: DecisionOutput
    escalation_status: EscalationOutput
