# rag_eval.py
import json
import torch
import numpy as np
import faiss
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer
from peft import PeftModel

# === Load Tokenizer and Fine-Tuned Model ===
base_model_path = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
lora_model_path = "./finetuned_rag_lora"

# Load tokenizer and apply special tokens
tokenizer = AutoTokenizer.from_pretrained(lora_model_path)
tokenizer.add_special_tokens({"additional_special_tokens": ["<|user|>", "<|assistant|>"]})
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Load base model and resize embeddings to match tokenizer
model = AutoModelForCausalLM.from_pretrained(
    base_model_path,
    torch_dtype=torch.float16,
    device_map="auto"
)
model.resize_token_embeddings(len(tokenizer))

# Load LoRA adapter on top of base model
model = PeftModel.from_pretrained(model, lora_model_path)
model.eval()

# === Load Retriever and FAISS Index ===
retriever_model = SentenceTransformer("all-MiniLM-L6-v2")
index = faiss.read_index("rag_index.faiss")
with open("rag_dataset_texts.json", "r", encoding="utf-8") as f:
    dataset_texts = json.load(f)

# === RAG Section-wise Generator ===
def retrieve_context(input_text, top_k=2):
    query_embedding = retriever_model.encode([input_text])[0]
    distances, indices = index.search(np.array([query_embedding], dtype=np.float32), top_k)
    return "\n\n".join([dataset_texts[i] for i in indices[0]])

# === Inference Configuration ===
sample_instruction = """event_title: Massive Wildfire Engulfs Northern California
event_description: A fast-moving wildfire fueled by dry winds and heat has spread across thousands of acres in Northern California, destroying homes and prompting large-scale evacuations.
disaster_type: Wildfire
event_date_time: 2025-08-10 14:30:00 UTC
event_location: Northern California, USA
event_coordinates: [-121.5, 39.5, 300.0]
disaster_details: fire_size: 85,000 acres, containment: 15%, wind_speed: 60 km/h, alert_level: Red, affected_population: 350,000, damage_reports: 1,200 structures destroyed, multiple highways closed, widespread smoke inhalation.
climate_data: temperature: 42.0°C, windspeed: 60 km/h, winddirection: 310° (northwest), humidity: 12.0%, precipitation_probability: 0%, cloud_cover: 10.0%, pressure_sea_level: 1001.2 hPa"""

system_prompt = (
    "You are a helpful assistant trained to provide detailed, structured disaster analysis reports. "
    "Output exactly these 7 sections with verbose, comprehensive information: "
    "<disaster_analysis>, <event_summary>, <detailed_analysis>, <predictions>, <impacts>, <mitigation_strategies>, <recommendations>"
)

sections = [
    "<disaster_analysis>",
    "<event_summary>",
    "<detailed_analysis>",
    "<predictions>",
    "<impacts>",
    "<mitigation_strategies>",
    "<recommendations>"
]

retrieved_context = retrieve_context(sample_instruction)
output_so_far = ""

print("\n=== Generating Disaster Report by Section ===\n")

for section in sections:
    prompt = f"""<|user|>
{system_prompt}

### Instruction:
Given the following input and retrieved context, generate ONLY the section: {section}

### Input:
{sample_instruction}

### Retrieved Context:
{retrieved_context}

<|assistant|>
{section}\n"""

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=1024,
            temperature=0.7,
            top_p=0.95,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.2,
        )
    generated = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Extract only new generation beyond the prompt
    generated_section = generated[len(prompt):].strip()

    # Stop at first next section tag
    for tag in sections:
        if tag in generated_section and tag != section:
            generated_section = generated_section.split(tag)[0].strip()
            break

    output_so_far += f"\n{section}\n{generated_section}\n"
    print(f"\n--- {section} ---\n{generated_section}\n")

with open("rag_finetuned_output.txt", "w", encoding="utf-8") as f:
    f.write(output_so_far)

print("\n✅ Full structured disaster report saved to rag_finetuned_output.txt")
