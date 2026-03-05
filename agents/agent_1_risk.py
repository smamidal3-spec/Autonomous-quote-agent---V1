import pandas as pd
import joblib
import shap
import json
import os
from agents.schema import QuoteInput, RiskOutput

class RiskProfilerAgent:
    def __init__(self, models_dir="models"):
        self.rf_model = joblib.load(os.path.join(models_dir, "risk_profiler_rf.pkl"))
        self.encoders = joblib.load(os.path.join(models_dir, "categorical_encoders.pkl"))
        self.feature_cols = joblib.load(os.path.join(models_dir, "feature_columns.pkl"))
        # Using a much lighter SHAP explainer for latency if needed, but TreeExplainer is fast
        self.explainer = shap.TreeExplainer(self.rf_model)

    def _strict_risk_heuristic(self, quote: QuoteInput) -> str:
        """Deterministic risk heuristic that ALWAYS overrides ML when triggered."""
        risk_points = 0
        
        # Accidents
        if quote.Prev_Accidents >= 2: risk_points += 3
        elif quote.Prev_Accidents == 1: risk_points += 1
        
        # Citations
        if quote.Prev_Citations >= 3: risk_points += 2
        elif quote.Prev_Citations >= 2: risk_points += 2
        elif quote.Prev_Citations >= 1: risk_points += 1
        
        # Driver Age
        if quote.Driver_Age < 25: risk_points += 2
        elif quote.Driver_Age > 65: risk_points += 1
        
        # Annual Miles
        if isinstance(quote.Annual_Miles_Range, str) and ">" in quote.Annual_Miles_Range and "15" in quote.Annual_Miles_Range:
            risk_points += 1
            
        if risk_points >= 4:
            return 'HIGH'
        elif risk_points >= 2:
            return 'MEDIUM'
        return 'LOW'

    def process(self, quote: QuoteInput) -> RiskOutput:
        # 1. Convert quote to dataframe row
        input_data = quote.model_dump()
        df = pd.DataFrame([input_data])
        
        # Add missing columns with median/mode to match exact training features
        for col in self.feature_cols:
            if col not in df.columns:
                df[col] = 0.0 # simplified default for missing fields
                
        # 2. Encode categorical variables
        for col in self.feature_cols:
            if col in self.encoders and col in df.columns:
                try:
                    df[col] = self.encoders[col].transform(df[col].astype(str))
                except ValueError:
                    df[col] = 0 # Default if unseen
                    
        # Ensure exact column order
        df = df[self.feature_cols]
        
        # 3. ML Prediction
        ml_prediction = self.rf_model.predict(df)[0]
        
        # 4. Deterministic Heuristic Override (strict business rules always win)
        heuristic_tier = self._strict_risk_heuristic(quote)
        
        # The heuristic UPGRADES risk but never downgrades it
        tier_order = {'LOW': 0, 'MEDIUM': 1, 'HIGH': 2}
        if tier_order.get(heuristic_tier, 0) > tier_order.get(ml_prediction, 0):
            prediction = heuristic_tier
        else:
            prediction = ml_prediction
        
        # Predict Probabilities to extract a 'risk score'
        probs = self.rf_model.predict_proba(df)[0]
        classes = self.rf_model.classes_
        prob_dict = dict(zip(classes, probs))
        
        # Calculate a 0-100 risk score based on high/medium probability
        risk_score = (prob_dict.get('HIGH', 0) * 100) + (prob_dict.get('MEDIUM', 0) * 50)
        
        # 5. Explainability (SHAP)
        shap_values = self.explainer.shap_values(df)
        
        # Get top feature pushing this prediction
        class_idx = list(classes).index(ml_prediction)
        
        import numpy as np
        if isinstance(shap_values, list):
            sv = shap_values[class_idx][0]
        else:
            shap_array = np.array(shap_values)
            if len(shap_array.shape) == 3:
                sv = shap_array[0, :, class_idx]
            else:
                sv = shap_array[0]
            
        sv = np.array(sv).flatten()
            
        # Find highest absolute impact feature
        sorted_indices = np.argsort(-np.abs(sv))
        top_feature = self.feature_cols[sorted_indices[0]]
        
        explanation = f"Classified as {prediction} risk primarily due to {top_feature}."

        return RiskOutput(
            risk_score=round(risk_score, 1),
            risk_tier=prediction,
            risk_explanation=explanation
        )
