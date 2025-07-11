# test_prompt_tuning.py
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

model_path = "./prompt_tuning_output"
base_model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
tokenizer = AutoTokenizer.from_pretrained(model_path)
base_model = AutoModelForCausalLM.from_pretrained(base_model_name, torch_dtype=torch.bfloat16)
model = PeftModel.from_pretrained(base_model, model_path)  # Load prompt-tuned config
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

prompt = """event_title: M 6.0 - 7 km W of Los Angeles, CA
event_description: Strong quake near city.
disaster_type: Earthquake
event_date_time: 2025-03-04 10:00:00 UTC
event_location: 7 km W of Los Angeles, CA
event_coordinates: [-118.2437, 34.0522, 12.0]
disaster_details: magnitude: 6.0, alert_level: Yellow, tsunami_risk: 0, significance_level: 600, felt_reports: 3000, damage_reports: Minor structural damage reported.
climate_data: temperature: 20.0°C, windspeed: 10.0 km/h, winddirection: 180°, humidity: 60.0%, precipitation_probability: 10%, cloud_cover: 30.0%, pressure_sea_level: 1014.0 hPa
Analysis:"""
inputs = tokenizer(prompt, return_tensors="pt").to(device)
outputs = model.generate(
    **inputs,
    max_new_tokens=600,
    temperature=0.5,
    top_p=0.95,
    do_sample=True
)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(f"Response: {response.split('Analysis:')[-1].strip() if 'Analysis:' in response else response}")