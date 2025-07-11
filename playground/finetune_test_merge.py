from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# 1. Load the finetuned merged model and tokenizer
model_path = "./finetuned_model_merged_100_1.5"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", torch_dtype=torch.float16)

# Ensure padding token is set
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = model.config.eos_token_id

# 2. Define the system prompt (same as training)
system_prompt = "You are a helpful assistant trained to provide detailed, structured disaster analysis reports without internal reasoning. Use the following structure: <disaster_analysis>, <event_summary>, <detailed_analysis>, <predictions>, <impacts>, <mitigation_strategies>."

# 3. Define a new instruction for testing
new_instruction = (
    "event_title: Tornado Outbreak Tears Through Oklahoma City\nevent_description: A swarm of tornadoes ripped across Oklahoma City, leveling homes and scattering debris.\ndisaster_type: Tornado Outbreak\nevent_date_time: 2025-05-15 19:00:00 UTC\nevent_location: Oklahoma City, Oklahoma, USA\nevent_coordinates: [-97.5, 35.5, 370.0]\ndisaster_details: count: 10 tornadoes, max_windspeed: 250 km/h, alert_level: Red, affected_population: 1,500,000, damage_reports: 8,000 homes destroyed, 150 dead, power lines down.\nclimate_data: temperature: 28.1°C, windspeed: 40 km/h, winddirection: 180° (southerly), humidity: 80.0%, precipitation_probability: 70%, cloud_cover: 90.0%, pressure_sea_level: 1002.3 hPa"
)

# 4. Prepare the input with a structure cue
input_text = f"<|user|>{system_prompt} {new_instruction}<|assistant|>"
# input_text = f"{new_instruction}<|assistant|><disaster_analysis>"
inputs = tokenizer(input_text, return_tensors="pt").to("cuda")

# 5. Generate the output
outputs = model.generate(
    **inputs,
    max_new_tokens=2048,
    do_sample=True,
    temperature=0.3,  # Lowered for more deterministic output
    top_p=0.9,
    pad_token_id=tokenizer.eos_token_id
)

# 6. Decode and print the result
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=False)
print("\nInference Output:")
print(generated_text.split("<|assistant|>")[1])