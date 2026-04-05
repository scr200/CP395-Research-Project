from google import genai
from google.genai import types
import json
import time

# 1. Setup API (Using the NEW SDK)
client = genai.Client(api_key="AIzaSyDqyx488F-h0LmYJXy2niOhVVJXrmVCj-k") # -- REPLACE APY KEY --

# 2. Define the Prompt
system_prompt = """
You are an AIOps Reliability Agent managing a cloud autoscaler. 
Analyze the provided anomaly log. Your goal is to classify the root cause and recommend an action.
You MUST output ONLY valid JSON using the following schema:
{
  "root_cause": "Brief description of why it failed",
  "recommended_action": "DECREASE_COOLDOWN" | "LOWER_REACTIVE_THRESHOLD" | "NO_ACTION",
  "new_fallback_threshold": integer,
  "new_cooldown": integer
}
"""

def evaluate_anomaly(log_entry):
    """Sends the log to the LLM and returns the parsed JSON."""
    user_prompt = f"Here is the anomaly log:\n{json.dumps(log_entry, indent=2)}"
    
    try:
        # New model and generation config syntax
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=system_prompt + "\n\n" + user_prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json"
            )
        )
        return json.loads(response.text)
    except Exception as e:
        return {"error": str(e)}

def apply_guardrails(llm_output, current_config):
    """Verifies the LLM output is safe to apply to the system."""
    safe_config = current_config.copy()
    
    # Check if LLM output is valid
    if "error" in llm_output or "new_fallback_threshold" not in llm_output:
        print("Guardrail Triggered: Invalid LLM Output. Discarding.")
        return safe_config
    
    # Guardrail 1: Threshold must be between 50 and 95
    suggested_threshold = llm_output["new_fallback_threshold"]
    if 50 <= suggested_threshold <= 95:
        safe_config["reactive_fallback_threshold"] = suggested_threshold
    else:
        print(f"Guardrail Triggered: Threshold {suggested_threshold} is unsafe. Ignoring.")
        
    # Guardrail 2: Action Whitelist
    valid_actions = ["DECREASE_COOLDOWN", "LOWER_REACTIVE_THRESHOLD", "NO_ACTION"]
    if llm_output.get("recommended_action") not in valid_actions:
        print(f"Guardrail Triggered: Unknown action {llm_output.get('recommended_action')}. Ignoring.")
        
    return safe_config

# 3. Run the Test
if __name__ == "__main__":
    # Load your test scenarios
    with open("/Users/sufiya/my_data_folder/classes/CP395/test_logs.json", "r") as f:
        scenarios = json.load(f)
        
    for log in scenarios:
        print(f"\n--- Processing Scenario {log['scenario_id']} ---")
        
        # Call the LLM
        llm_decision = evaluate_anomaly(log)
        print("LLM Output:", json.dumps(llm_decision, indent=2))
        
        # Apply Guardrails
        final_config = apply_guardrails(llm_decision, log["current_config"])
        print("Final Safe Configuration:", final_config)

        # Apply Guardrails
        final_config = apply_guardrails(llm_decision, log["current_config"])
        print("Final Safe Configuration:", final_config)
        
        # Pause for 15 seconds to avoid the Free Tier rate limit
        time.sleep(15)
