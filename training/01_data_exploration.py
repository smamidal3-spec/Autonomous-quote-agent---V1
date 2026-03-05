import pandas as pd
import os

data_path = "quote_agents/data/use_case_03/USE CASE - 03/Autonomous QUOTE AGENTS.csv"
print(f"Loading dataset from: {data_path}")

df = pd.read_csv(data_path)

print("\n--- Dataset Info ---")
df.info()

print("\n--- Missing Values ---")
print(df.isnull().sum())

print("\n--- Target Variables ---")
print("1. Target for Risk Profiler (Risk_Tier - assumes it exists or we need to derive it)")
if 'Risk_Tier' in df.columns:
    print(df['Risk_Tier'].value_counts(normalize=True))
else:
    print("Risk_Tier column not found - we may need to engineer this based on accidents/citations.")

print("\n2. Target for Conversion Predictor (converted)")
if 'converted' in df.columns:
    print(df['converted'].value_counts(normalize=True))
elif 'Converted' in df.columns:
    print(df['Converted'].value_counts(normalize=True))
else:
    print("Conversion column not found. Let's look at numerical/categorical features.")

print("\n--- First 5 rows ---")
pd.set_option('display.max_columns', None)
print(df.head())
