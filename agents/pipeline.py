import uuid
from agents.schema import QuoteInput, PipelineOutput
from agents.agent_1_risk import RiskProfilerAgent
from agents.agent_2_conversion import ConversionPredictorAgent
from agents.agent_3_premium import PremiumAdvisorAgent
from agents.agent_4_router import DecisionRouterAgent

class MultiAgentPipeline:
    def __init__(self, models_dir="models"):
        # Initialize the 4 agents
        self.agent_1 = RiskProfilerAgent(models_dir)
        self.agent_2 = ConversionPredictorAgent(models_dir)
        self.agent_3 = PremiumAdvisorAgent()
        self.agent_4 = DecisionRouterAgent()
        
    def execute(self, quote: QuoteInput) -> PipelineOutput:
        # Step 1: Risk Profiling
        risk_out = self.agent_1.process(quote)
        
        # Step 2: Conversion Prediction
        conversion_out = self.agent_2.process(quote, risk_out.risk_tier)
        
        # Step 3: Premium Advice
        premium_out = self.agent_3.process(quote, conversion_out.conversion_probability, risk_out.risk_tier)
        
        # Step 4: Decision Routing & Escalation
        decision_out, escalation_out = self.agent_4.process(risk_out, conversion_out, premium_out)
        
        # Return structured JSON Pipeline Output
        return PipelineOutput(
            quote_id=str(uuid.uuid4()),
            risk_evaluation=risk_out,
            conversion_prediction=conversion_out,
            premium_advice=premium_out,
            final_decision=decision_out,
            escalation_status=escalation_out
        )
