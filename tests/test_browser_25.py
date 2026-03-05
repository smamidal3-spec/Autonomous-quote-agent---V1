import json
import urllib.parse
from playwright.sync_api import sync_playwright

def run_ui_tests():
    with open("tests/synthetic_data.json", "r") as f:
        profiles = json.load(f)
        
    print(f"\n🌐 Starting Automated Browser UI Tests for {len(profiles)} Profiles...\n")
    correct = 0
    total = len(profiles)
    
    with sync_playwright() as p:
        # Launch headless chromium
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        
        for i, profile in enumerate(profiles):
            name = profile['Profile_Name']
            
            # Construct URL parameters matching the exact profile
            params = {
                "Prev_Accidents": profile["Prev_Accidents"],
                "Prev_Citations": profile["Prev_Citations"],
                "Driving_Exp": profile["Driving_Exp"],
                "Driver_Age": profile["Driver_Age"],
                "Veh_Usage": profile["Veh_Usage"],
                "Annual_Miles_Range": profile["Annual_Miles_Range"],
                "Q_Valid_DT": profile["Q_Valid_DT"],
                "Coverage": profile["Coverage"],
                "Agent_Type": profile["Agent_Type"],
                "Region": profile["Region"],
                "Sal_Range": profile["Sal_Range"],
                "HH_Drivers": profile["HH_Drivers"],
                "Re_Quote": profile["Re_Quote"],
                "Vehicl_Cost_Range": profile["Vehicl_Cost_Range"],
                "Quoted_Premium": profile["Quoted_Premium"]
            }
            query_string = urllib.parse.urlencode(params)
            url = f"http://localhost:8501/?{query_string}"
            
            # Navigate to the Streamlit app with inputs pre-filled via URL
            page.goto(url)
            
            # Wait for Streamlit to render the run button
            run_btn = page.locator("button:has-text('Run Strict JSON Pipeline')")
            run_btn.wait_for(state="visible", timeout=10000)
            
            # Click to execute the 4-Agent pipeline
            run_btn.click()
            
            # Wait for the results to process and render
            # We know it outputs "Quote ID:" when done
            page.locator("text=Quote ID:").wait_for(state="visible", timeout=15000)
            
            # Extract the pure JSON from the hidden div
            raw_div = page.locator("#raw-testing-json")
            raw_div.wait_for(state="attached", timeout=15000)
            json_str = raw_div.inner_text()
            
            # Use strict python JSON parsing for test validation
            try:
                res = json.loads(json_str)
                decision = res["final_decision"]["decision"]
                tier = res["risk_evaluation"]["risk_tier"]
                prem = res["premium_advice"]["premium_issue"]
                esc = res["escalation_status"]["escalation_required"]
                
                # Verify Accuracy from UI results strictly
                is_accurate = False
                
                if "Low_Risk" in name:
                    if esc == False and tier in ["LOW", "MEDIUM"]:
                        is_accurate = True
                elif "High_Risk" in name:
                    if tier == "HIGH" and decision == "ESCALATE_TO_UNDERWRITER":
                        is_accurate = True
                elif "Medium_Risk" in name:
                    if tier in ["MEDIUM", "HIGH"]: 
                        is_accurate = True
                elif "Premium_Sensitive" in name:
                    if prem == True:
                        is_accurate = True
                elif "Escalation_Guarantee" in name:
                    if esc == True:
                        is_accurate = True
            except Exception as e:
                is_accurate = False
                tier = "PARSE_ERROR"
                decision = str(e)
                    
            if is_accurate:
                correct += 1
                print(f"[{i+1}/{total}] {name}: ✅ UI VERIFIED")
            else:
                print(f"[{i+1}/{total}] {name}: ❌ UI LOGIC MISMATCH (Tier: {tier}, Decision: {decision})")
                
        browser.close()
        
    print("\n==================================================")
    print(f"🎯 Automated Browser UI Accuracy: {correct}/{total} ({(correct/total)*100:.1f}%)")
    print("==================================================")

if __name__ == "__main__":
    run_ui_tests()
