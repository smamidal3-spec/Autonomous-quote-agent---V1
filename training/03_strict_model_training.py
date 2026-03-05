import pandas as pd
import numpy as np
import os
import joblib
import json
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
from imblearn.over_sampling import SMOTE

print("--- Starting Strict Model Training ---")

# 1. Load Data
data_path = "data/use_case_03/USE CASE - 03/Autonomous QUOTE AGENTS.csv"
if not os.path.exists(data_path):
    print(f"Error: Data not found at {data_path}. Run from quote_agents root.")
    exit(1)

df = pd.read_csv(data_path)
print(f"Loaded dataset with {len(df)} rows.")

# 2. Strict Feature Engineering
def strict_risk_heuristic(row):
    """
    Deterministic risk logic per strict architecture:
    High risk when: many accidents, young driver, high annual mileage, many citations.
    """
    risk_points = 0
    
    # Accidents
    accidents = row.get('Prev_Accidents')
    if pd.notna(accidents):
        if accidents >= 2: risk_points += 3
        elif accidents == 1: risk_points += 1
        
    # Citations
    citations = row.get('Prev_Citations')
    if pd.notna(citations):
        if citations >= 3: risk_points += 2
        elif citations >= 1: risk_points += 1
        
    # Driver Age
    age = row.get('Driver_Age')
    if pd.notna(age):
        if age < 25: risk_points += 2
        elif age > 65: risk_points += 1
        
    # Annual Miles
    miles = row.get('Annual_Miles_Range')
    if pd.notna(miles) and isinstance(miles, str):
        if ">" in miles and "15" in miles: # e.g. "> 15 K"
            risk_points += 1
            
    if risk_points >= 4:
        return 'HIGH'
    elif risk_points >= 2:
        return 'MEDIUM'
    return 'LOW'

df['Risk_Tier_Target'] = df.apply(strict_risk_heuristic, axis=1)

# Target 2: Conversion
df['Conversion_Target'] = df['Policy_Bind'].apply(lambda x: 1 if x == 'Yes' else 0)

# Build Features
drop_cols = ['Quote_Num', 'Policy_Bind', 'Risk_Tier_Target', 'Conversion_Target']
features = df.drop(columns=[c for c in drop_cols if c in df.columns], errors='ignore')

# Handle Categorical Variables using Label Encoding
encoders = {}
for col in features.columns:
    if features[col].dtype == 'object':
        le = LabelEncoder()
        features[col] = features[col].astype(str)
        features[col] = le.fit_transform(features[col])
        encoders[col] = le

# Handle Missing Values
features = features.fillna(features.median(numeric_only=True))

print(f"Engineered targets. Risk distribution:\n{df['Risk_Tier_Target'].value_counts(normalize=True)}")
print(f"Conversion rate:\n{df['Conversion_Target'].value_counts(normalize=True)}")

# 3. Train Agent 1: Risk Profiler (Random Forest)
print("\n--- Training Agent 1: Risk Profiler ---")
X = features
y_risk = df['Risk_Tier_Target']

X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(X, y_risk, test_size=0.2, random_state=42)

rf_model = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42, class_weight='balanced')
rf_model.fit(X_train_r, y_train_r)
print(f"Risk Profiler Accuracy: {rf_model.score(X_test_r, y_test_r):.2f}")

# 4. Train Agent 2: Conversion Predictor (XGBoost with SMOTE)
print("\n--- Training Agent 2: Conversion Predictor ---")
X_conv = features.copy()
risk_encoder = LabelEncoder()
X_conv['Risk_Tier'] = risk_encoder.fit_transform(df['Risk_Tier_Target'])
y_conv = df['Conversion_Target']

X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(X_conv, y_conv, test_size=0.2, random_state=42)

# Handle Class Imbalance using SMOTE
smote = SMOTE(random_state=42)
X_train_sm, y_train_sm = smote.fit_resample(X_train_c, y_train_c)

xgb_model = xgb.XGBClassifier(n_estimators=150, max_depth=4, learning_rate=0.05, random_state=42, use_label_encoder=False, eval_metric='logloss')
xgb_model.fit(X_train_sm, y_train_sm)
print(f"Conversion Predictor Accuracy: {xgb_model.score(X_test_c, y_test_c):.2f}")

# 5. Save Models
os.makedirs("models", exist_ok=True)
joblib.dump(rf_model, "models/risk_profiler_rf.pkl")
joblib.dump(xgb_model, "models/conversion_predictor_xgb.pkl")
joblib.dump(encoders, "models/categorical_encoders.pkl")
joblib.dump(risk_encoder, "models/risk_encoder.pkl")
joblib.dump(list(X.columns), "models/feature_columns.pkl")
print("Models saved successfully.")

# 6. Generate Synthetic Test Dataset (25 Profiles)
print("\n--- Generating Synthetic Test Profiles ---")
os.makedirs("tests", exist_ok=True)

synthetic_profiles = []
base_profile = {
    "Agent_Type": "EA", "Q_Creation_DT": "2019/10/01", "Q_Valid_DT": "2019/11/29", 
    "Policy_Bind_DT": "2019/10/02", "Region": "A", "Agent_Num": 10.0, "Policy_Type": "Truck", 
    "HH_Vehicles": 1.0, "HH_Drivers": 1.0, "Driver_Age": 35.0, "Driving_Exp": 10.0, 
    "Prev_Accidents": 0.0, "Prev_Citations": 0.0, "Gender": "Male", "Marital_Status": "Married", 
    "Education": "Bachelors", "Sal_Range": "50 K - 75 K", "Coverage": "Balanced", 
    "Veh_Usage": "Business", "Annual_Miles_Range": "<= 7.5 K", "Vehicl_Cost_Range": "10 K - 20 K", 
    "Re_Quote": "No", "Quoted_Premium": 500.0, "Proposed_Premium": 500.0
}

# Profile 1-5: Low Risk / High Conversion
for i in range(5):
    p = base_profile.copy()
    p['Prev_Accidents'] = 0.0
    p['Driver_Age'] = 30 + i
    p['Quoted_Premium'] = 300.0 + (i*10)
    p['Proposed_Premium'] = p['Quoted_Premium']
    p['Profile_Name'] = f"Low_Risk_High_Conv_{i+1}"
    synthetic_profiles.append(p)

# Profile 6-10: High Risk (Accidents + Young)
for i in range(5):
    p = base_profile.copy()
    p['Prev_Accidents'] = float(2 + i)
    p['Prev_Citations'] = float(1 + i)
    p['Driver_Age'] = 20 + i
    p['Profile_Name'] = f"High_Risk_{i+1}"
    synthetic_profiles.append(p)

# Profile 11-15: Medium Risk (Some citations)
for i in range(5):
    p = base_profile.copy()
    p['Prev_Citations'] = 2.0
    p['Driver_Age'] = 40 + i
    p['Profile_Name'] = f"Medium_Risk_{i+1}"
    synthetic_profiles.append(p)

# Profile 16-20: Premium Sensitive (Low Salary, High Premium)
for i in range(5):
    p = base_profile.copy()
    p['Sal_Range'] = "<= $ 25 K"
    p['Quoted_Premium'] = 2000.0 + (i*100)
    p['Proposed_Premium'] = p['Quoted_Premium']
    p['Profile_Name'] = f"Premium_Sensitive_{i+1}"
    synthetic_profiles.append(p)

# Profile 21-25: High Risk + Premium Sensitive (Escalation guarantee)
for i in range(5):
    p = base_profile.copy()
    p['Prev_Accidents'] = 3.0
    p['Driver_Age'] = 19.0
    p['Sal_Range'] = "<= $ 25 K"
    p['Quoted_Premium'] = 3500.0
    p['Proposed_Premium'] = 3500.0
    p['Profile_Name'] = f"Escalation_Guarantee_{i+1}"
    synthetic_profiles.append(p)

with open("tests/synthetic_data.json", "w") as f:
    json.dump(synthetic_profiles, f, indent=4)
    
print("Saved 25 synthetic profiles to tests/synthetic_data.json")
print("✅ Strict Data Pipeline Complete.")
