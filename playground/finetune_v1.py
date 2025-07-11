import json
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from peft import LoraConfig, get_peft_model
from datasets import Dataset
import torch

# 1. Load the model and tokenizer
model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", torch_dtype=torch.float16)

# Ensure padding token is set
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = model.config.eos_token_id

# 2. Prepare the dataset
system_prompt = "You are a helpful assistant trained to provide detailed, structured disaster analysis reports without internal reasoning. Use the following structure: <disaster_analysis>, <event_summary>, <detailed_analysis>, <predictions>, <impacts>, <mitigation_strategies>."
dataset_path = "disaster_data.jsonl"

# Read and format the JSONL data with error handling
formatted_data = []
with open(dataset_path, "r") as f:
    for i, line in enumerate(f, 1):
        try:
            sample = json.loads(line.strip())
            full_text = f"<|user|>{system_prompt} {sample['instruction']}<|assistant|>{sample['output']}"
            formatted_data.append({"text": full_text})
        except json.JSONDecodeError as e:
            print(f"Error in line {i}: {line.strip()}")
            print(f"Error message: {e}")
            raise

# Convert to Hugging Face Dataset
dataset = Dataset.from_list(formatted_data)

# Tokenize the dataset
def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=1024)  # Increased to 1024

tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["text"])

# 3. Set up LoRA configuration
lora_config = LoraConfig(
    r=16,
    lora_alpha=16,
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj"
    ],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

# Apply LoRA to the model
model = get_peft_model(model, lora_config)

# 4. Define training arguments
training_args = TrainingArguments(
    output_dir="./finetuned_model",
    per_device_train_batch_size=1,
    num_train_epochs=5,
    learning_rate=2e-4,
    warmup_steps=5,
    weight_decay=0.01,
    logging_steps=1,
    save_strategy="epoch",
    fp16=True,
    gradient_accumulation_steps=4,
)

# 5. Set up data collator
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# 6. Initialize and run the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=data_collator,
)

print("Starting training...")
trainer.train()
print("Training completed.")

# 7. Merge LoRA weights and save the model
model = model.merge_and_unload()
model.save_pretrained("./finetuned_model_merged")
tokenizer.save_pretrained("./finetuned_model_merged")
print("Model saved to './finetuned_model_merged'")

# 8. Inference example with a structured prompt
new_instruction = (
    "event_title: M 4.8 - 15 km SE of Lima, Peru\n"
    "event_description: A minor earthquake occurred near Lima, Peru, with slight shaking reported.\n"
    "disaster_type: Earthquake\n"
    "event_date_time: 2025-03-20 09:15:30 UTC\n"
    "event_location: 15 km SE of Lima, Peru\n"
    "event_coordinates: [-76.9876, -12.1234, 10.5]\n"
    "disaster_details: magnitude: 4.8, alert_level: Green, tsunami_risk: 0, significance_level: 280, felt_reports: 800, damage_reports: No significant damage.\n"
    "climate_data: temperature: 25.0°C, windspeed: 10.5 km/h, winddirection: 180°, humidity: 65%, precipitation_probability: 10%, cloud_cover: 30%, pressure_sea_level: 1015.0 hPa\n"
    "Provide a detailed analysis with the following structure: <disaster_analysis>, <event_summary>, <detailed_analysis>, <predictions>, <impacts>, <mitigation_strategies>."
)

input_text = f"<|user|>{system_prompt} {new_instruction}<|assistant|>"
inputs = tokenizer(input_text, return_tensors="pt").to("cuda")

outputs = model.generate(
    **inputs,
    max_new_tokens=4096,  # Increased to 1024 for full output
    do_sample=True,
    temperature=0.7,
    top_p=0.95,
    pad_token_id=tokenizer.eos_token_id
)

generated_text = tokenizer.decode(outputs[0], skip_special_tokens=False)
print("\nInference Output:")
print(generated_text.split("<|assistant|>")[1])
