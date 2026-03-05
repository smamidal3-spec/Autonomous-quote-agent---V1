from agents.schema import QuoteInput, PremiumOutput

class PremiumAdvisorAgent:
    def process(self, quote: QuoteInput, conversion_probability: float, risk_tier: str) -> PremiumOutput:
        issue = False
        reasons = []
        min_p = quote.Quoted_Premium
        max_p = quote.Quoted_Premium
        
        # Logic 1: High Premium relative to Salary
        if "25 K" in str(quote.Sal_Range) and quote.Quoted_Premium > 1500:
            issue = True
            reasons.append("Premium above expected range for salary")
            min_p = quote.Quoted_Premium * 0.70
            max_p = quote.Quoted_Premium * 0.85
            
        # Logic 2: Low Conversion despite Good Risk
        elif conversion_probability < 30 and risk_tier == "LOW":
            issue = True
            reasons.append("Low conversion despite low risk profile")
            min_p = quote.Quoted_Premium * 0.80
            max_p = quote.Quoted_Premium * 0.90
            
        # Logic 3: High Risk demands higher premium
        elif risk_tier == "HIGH":
            # We don't discount high risk
            min_p = quote.Quoted_Premium * 1.10
            max_p = quote.Quoted_Premium * 1.30
            
        reason_str = " | ".join(reasons) if reasons else "Premium is competitive and appropriate."
        
        return PremiumOutput(
            premium_issue=issue,
            recommended_premium_range=[round(min_p, 2), round(max_p, 2)],
            recommendation_reason=reason_str
        )
