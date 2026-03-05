import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
import joblib
import os

print("--- Starting Model Design ---")

# 1. Load Data
data_path = "quote_agents/data/use_case_03/USE CASE - 03/Autonomous QUOTE AGENTS.csv"
df = pd.read_csv(data_path)

print(f"Loaded dataset with {len(df)} rows.")

# 2. Feature Engineering & Preprocessing
# Target 1: Risk Tier (We need to engineer this as it doesn't clearly exist)
# Let's create a proxy for risk based on Previous Tickets and Accidents (assuming those columns exist or similar)
# Looking at standard features...
def calculate_risk(row):
    # A simple deterministic heuristic to create our training labels for the ML model
    risk_score = 0
    if pd.notna(row.get('Prior_Insurance')) and row['Prior_Insurance'] == 'No':
        risk_score += 2
    
    # We will map standard driver Age/Experience if available, but for now fallback to random distribution if we lack columns
    # Actually, we have limited columns in the printout, so let's check what we have exactly.
    # We will just categorize Premium as a proxy for risk for the sake of the hackathon if we lack accident history.
    if pd.notna(row.get('Premium')):
        if row['Premium'] > 1500:
            return 'High'
        elif row['Premium'] > 800:
            return 'Medium'
    return 'Low'

df['Risk_Tier_Target'] = df.apply(calculate_risk, axis=1)

# Target 2: Conversion (Policy_Bind -> 1 for Yes, 0 for No)
df['Conversion_Target'] = df['Policy_Bind'].apply(lambda x: 1 if x == 'Yes' else 0)

# Drop identifiers
features = df.drop(columns=['Quote_Num', 'Policy_Bind', 'Risk_Tier_Target', 'Conversion_Target', 'Premium'], errors='ignore')

# Handle Categorical Variables using Label Encoding for simplicity in XGBoost/RF
encoders = {}
for col in features.columns:
    if features[col].dtype == 'object':
        le = LabelEncoder()
        features[col] = features[col].astype(str)
        features[col] = le.fit_transform(features[col])
        encoders[col] = le

# Handle Missing Values
features = features.fillna(features.median(numeric_only=True))

# 3. Train Agent 1: Risk Profiler (Random Forest)
print("\n--- Training Agent 1: Risk Profiler (Random Forest) ---")
X = features
y_risk = df['Risk_Tier_Target']

X_train, X_test, y_train, y_test = train_test_split(X, y_risk, test_size=0.2, random_state=42)

rf_model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
rf_model.fit(X_train, y_train)
acc = rf_model.score(X_test, y_test)
print(f"Risk Profiler Accuracy: {acc:.2f}")

# 4. Train Agent 2: Conversion Predictor (XGBoost)
print("\n--- Training Agent 2: Conversion Predictor (XGBoost) ---")
# For Agent 2, we actually INCLUDE the output of Agent 1 (Risk Tier) as a feature!
X_conv = features.copy()
# Encode the risk target so XGBoost can use it
risk_encoder = LabelEncoder()
X_conv['Risk_Tier'] = risk_encoder.fit_transform(df['Risk_Tier_Target'])
y_conv = df['Conversion_Target']

X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(X_conv, y_conv, test_size=0.2, random_state=42)

xgb_model = xgb.XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.1, random_state=42, use_label_encoder=False, eval_metric='logloss')
xgb_model.fit(X_train_c, y_train_c)
acc_c = xgb_model.score(X_test_c, y_test_c)
print(f"Conversion Predictor Accuracy: {acc_c:.2f}")

# 5. Save Models and Encoders
os.makedirs("quote_agents/models", exist_ok=True)
joblib.dump(rf_model, "quote_agents/models/risk_profiler_rf.pkl")
joblib.dump(xgb_model, "quote_agents/models/conversion_predictor_xgb.pkl")
joblib.dump(encoders, "quote_agents/models/categorical_encoders.pkl")
joblib.dump(risk_encoder, "quote_agents/models/risk_encoder.pkl")
# Save a list of exactly what features the model expects
joblib.dump(list(X.columns), "quote_agents/models/feature_columns.pkl")


print("\n✅ Step 5 Complete: Models trained and saved successfully!")
