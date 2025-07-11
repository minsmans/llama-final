# rag_train_sectionwise.py
import os
import json
import torch
import gc
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from trl import SFTTrainer
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# === Configuration ===
model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
data_path = "dataset.jsonl"
output_dir = "./finetuned_rag_lora"

# === Load raw dataset ===
raw_dataset = load_dataset("json", data_files=data_path, split="train")

# === Expand dataset into section-wise samples ===
section_tags = [
    "<disaster_analysis>",
    "<event_summary>",
    "<detailed_analysis>",
    "<predictions>",
    "<impacts>",
    "<mitigation_strategies>",
    "<recommendations>"
]

expanded_samples = []
system_prompt = (
    "You are a helpful assistant trained to provide detailed, structured disaster analysis reports.\n"
    "Output exactly one of the 7 sections using structured text starting with a section header."
)

for example in raw_dataset:
    full_output = example["output"]
    for tag in section_tags:
        if tag in full_output:
            section_text = full_output.split(tag)[1].split("</")[0].strip()
            prompt = f"""<|user|>
{system_prompt}

### Instruction:
Generate ONLY the section: {tag}

### Input:
{example['instruction']}

<|assistant|>
{tag}\n{section_text}"""
            expanded_samples.append({"text": prompt})

# === Convert to HuggingFace Dataset ===
dataset = Dataset.from_list(expanded_samples).train_test_split(test_size=0.1, seed=42)

# === Load tokenizer and model ===
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.add_special_tokens({"additional_special_tokens": ["<|user|>", "<|assistant|>"]})
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto",
    load_in_4bit=True
)
model.resize_token_embeddings(len(tokenizer))
model = prepare_model_for_kbit_training(model)

# === Apply LoRA ===
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, lora_config)

# === Training Arguments ===
training_args = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    learning_rate=3e-4,
    num_train_epochs=8,
    logging_steps=10,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    fp16=True,
    optim="paged_adamw_8bit",
    lr_scheduler_type="cosine",
    warmup_steps=20,
    report_to="none",
    seed=3407
)

# === Train ===
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    args=training_args,
    formatting_func=lambda x: x["text"]
)

gc.collect()
torch.cuda.empty_cache()
trainer.train()

# === Save model ===
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)
print(f"\n✅ Fine-tuned model saved to {output_dir}")
