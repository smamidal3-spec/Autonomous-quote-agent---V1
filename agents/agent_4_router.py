from agents.schema import RiskOutput, ConversionOutput, PremiumOutput, DecisionOutput, EscalationOutput, RoutingExplanation

class DecisionRouterAgent:
    def process(self, risk: RiskOutput, conversion: ConversionOutput, premium: PremiumOutput) -> tuple[DecisionOutput, EscalationOutput]:
        
        confidence = 1.0
        decision = "UNKNOWN"
        reason = []
        
        risk_f = f"Risk Tier is {risk.risk_tier}"
        conv_f = f"Conversion Probability is {conversion.conversion_probability}%"
        prem_f = "Premium issue detected" if premium.premium_issue else "Premium is fine"
        
        explanation = RoutingExplanation(
            risk_factor=risk_f,
            conversion_driver=conv_f,
            premium_issue=prem_f
        )
        
        # Scenario 1: Escalate
        # High risk + low confidence OR Human Escalation Protocol
        if risk.risk_tier == "HIGH" or conversion.conversion_probability < 20:
            decision = "ESCALATE_TO_UNDERWRITER"
            confidence = 0.4
            reason.append("High risk profile or extremely low conversion.")
            
        # Scenario 2: Agent Follow-Up
        # Medium conversion + premium issue
        elif conversion.conversion_band == "MEDIUM" or premium.premium_issue:
            decision = "FOLLOW_UP_AGENT"
            confidence = 0.75
            reason.append("Borderline conversion or premium adjustment needed. Agent intervention required.")
            
        # Scenario 3: Auto Approve
        # LOW risk + HIGH conversion
        elif risk.risk_tier == "LOW" and conversion.conversion_band == "HIGH":
            decision = "AUTO_APPROVE"
            confidence = 0.95
            reason.append("Low risk and high conversion. Safe to auto-approve.")
            
        else:
            decision = "FOLLOW_UP_AGENT"
            confidence = 0.6
            reason.append("Fallback to agent review.")
            
        decision_out = DecisionOutput(
            decision=decision,
            confidence_score=confidence,
            decision_explanation=" ".join(reason),
            detailed_explanation=explanation
        )
        
        # Human Escalation Protocol
        escalation_req = False
        esc_reason = "None"
        
        if confidence < 0.6:
            escalation_req = True
            esc_reason = "System confidence dropped below 0.6 threshold."
        elif risk.risk_tier == "HIGH":
            escalation_req = True
            esc_reason = "High risk profile mandates escalation."
        elif risk.risk_tier == "MEDIUM" and premium.premium_issue:
            escalation_req = True
            esc_reason = "Medium risk with premium discrepancy triggers escalation."
            
        escalation_out = EscalationOutput(
            escalation_required=escalation_req,
            reason=esc_reason
        )
            
        return decision_out, escalation_out
