from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList
from peft import PeftModel
import torch

class StopOnTextCriteria(StoppingCriteria):
    def __init__(self, stop_text, tokenizer):
        self.stop_text = stop_text
        self.tokenizer = tokenizer
    def __call__(self, input_ids, scores, **kwargs):
        decoded_text = self.tokenizer.decode(input_ids[0], skip_special_tokens=False)
        if self.stop_text in decoded_text:
            print(f"Stopped at: {self.stop_text}")
            return True
        return False

model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", torch_dtype=torch.float16)
model = PeftModel.from_pretrained(model, "./finetuned_model_lora").to("cuda")

system_prompt = "You are a helpful assistant trained to provide detailed, structured disaster analysis reports. Output exactly these 7 sections without duplication or extra content, with verbose, comprehensive details matching the depth of the training examples: <disaster_analysis>, <event_summary>, <detailed_analysis>, <predictions>, <impacts>, <mitigation_strategies>, <recommendations>."
new_instruction = "event_title: Flash Floods Submerge Central Texas\nevent_description: Heavy rainfall from a stalled cold front has caused flash flooding across Central Texas, overwhelming drainage systems, submerging homes, and stranding residents in rapidly rising waters.\ndisaster_type: Flood\nevent_date_time: 2025-03-18 04:45:00 UTC\nevent_location: Central Texas, USA\nevent_coordinates: [-97.7, 30.3, 0.0]\ndisaster_details: rainfall_amount: 250 mm in 12 hours, water_depth: 1.5 m in low-lying areas, alert_level: Red, affected_population: 200,000, damage_reports: Homes flooded with water up to 1 meter indoors, highways closed including I-35, vehicles swept away by currents, emergency rescues in progress.\nclimate_data: temperature: 18.5°C, windspeed: 20 km/h, winddirection: 90° (easterly), humidity: 92.0%, precipitation_probability: 95%, cloud_cover: 100.0%, pressure_sea_level: 1002.3 hPa"
input_text = f"<|user|>{system_prompt} {new_instruction}<|assistant|><disaster_analysis>\n1. **Key Data Points Extraction:**"
inputs = tokenizer(input_text, return_tensors="pt").to("cuda")

stopping_criteria = StoppingCriteriaList([StopOnTextCriteria("</recommendations>", tokenizer)])
outputs = model.generate(
    **inputs,
    max_new_tokens=1800,  # Increased to match dataset length
    do_sample=True,
    temperature=0.8,  # Slightly higher for creativity
    top_p=0.95,  # Relaxed for more detail
    repetition_penalty=1.2,  # Penalize repetition
    pad_token_id=tokenizer.eos_token_id,
    stopping_criteria=stopping_criteria
)

generated_text = tokenizer.decode(outputs[0], skip_special_tokens=False)
print("\nInference Output:")
print(generated_text.split("<|assistant|>")[1])