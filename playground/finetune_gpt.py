import json
import torch
import os
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from peft import LoraConfig, get_peft_model, PeftModel
from datasets import Dataset
import gc

# Set environment variable to reduce memory fragmentation
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Load the dataset with error handling
def load_dataset(file_path, chunk_size=100):
    data = []
    chunk = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                chunk.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"Warning: Skipping malformed JSON at line {i}: {e}")
                continue
            if len(chunk) >= chunk_size:
                data.extend(chunk)
                chunk = []
                gc.collect()
        if chunk:
            data.extend(chunk)
    return data

# Function to split long sequences into chunks
def split_into_chunks(token_ids, max_length, overlap=20):
    chunks = []
    start = 0
    while start < len(token_ids):
        end = min(start + max_length, len(token_ids))
        chunk = token_ids[start:end]
        chunks.append(chunk)
        start = max(0, end - overlap) if end < len(token_ids) else end
    return chunks

# Prepare the dataset
data = load_dataset("dataset.jsonl", chunk_size=100)
system_prompt = (
    "You are a helpful assistant trained to provide detailed, structured disaster analysis reports. "
    "Output exactly these 7 sections without duplication or extra content, using structured text with angle-bracket headers "
    "(e.g., <disaster_analysis>, <event_summary>, etc.), with verbose, comprehensive details (1500–2000 tokens) matching the depth of the training examples: "
    "<disaster_analysis> (include 13 subsections like Key Data Points Extraction, Historical Patterns, etc.), "
    "<event_summary>, <detailed_analysis>, <predictions>, <impacts>, <mitigation_strategies>, <recommendations>. "
    "Do not output JSON; use structured text only."
)

# Load the tokenizer
model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Add special tokens, including angle-bracket headers
section_headers = [
    "<disaster_analysis>", "</disaster_analysis>",
    "<event_summary>", "</event_summary>",
    "<detailed_analysis>", "</detailed_analysis>",
    "<predictions>", "</predictions>",
    "<impacts>", "</impacts>",
    "<mitigation_strategies>", "</mitigation_strategies>",
    "<recommendations>", "</recommendations>"
]
special_tokens = {
    "additional_special_tokens": ["<|user|>", "<|assistant|>"] + section_headers
}
num_added_tokens = tokenizer.add_special_tokens(special_tokens)
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    tokenizer.pad_token = "[PAD]"

# Format the data for training with chunking
formatted_data = []
max_chunk_length = 512  # Reduced from 1024 to 512 to lower memory usage
for item in data:
    instruction = item["instruction"]
    output = item["output"]
    user_tokens = tokenizer.encode("<|user|>" + system_prompt + " " + instruction, add_special_tokens=False)
    assistant_tokens = tokenizer.encode("<|assistant|>" + output, add_special_tokens=False)
    eos_tokens = tokenizer.encode(tokenizer.eos_token, add_special_tokens=False)
    token_ids = user_tokens + assistant_tokens + eos_tokens
    chunks = split_into_chunks(token_ids, max_chunk_length - 10, overlap=20)
    for chunk in chunks:
        formatted_data.append({"input_ids": chunk, "labels": chunk.copy()})
    del token_ids, chunks
    gc.collect()

# Convert to Hugging Face Dataset
dataset = Dataset.from_list(formatted_data)
del formatted_data
gc.collect()

# Load the DeepSeek model
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto"
)
if num_added_tokens > 0 or tokenizer.pad_token == "[PAD]":
    model.resize_token_embeddings(len(tokenizer))
model.gradient_checkpointing_enable()

# Tokenize the dataset
def tokenize_function(examples):
    input_ids = examples["input_ids"]
    labels = examples["labels"]
    padded_input_ids = [
        ids + [tokenizer.pad_token_id] * (max_chunk_length - len(ids))
        for ids in input_ids
    ]
    padded_labels = [
        ids + [-100] * (max_chunk_length - len(ids))
        for ids in labels
    ]
    attention_mask = [
        [1] * len(ids) + [0] * (max_chunk_length - len(ids))
        for ids in input_ids
    ]
    return {
        "input_ids": padded_input_ids,
        "attention_mask": attention_mask,
        "labels": padded_labels
    }

tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["input_ids", "labels"])
del dataset
gc.collect()

# Split into train and validation sets
train_test_split = tokenized_dataset.train_test_split(test_size=0.2)
train_dataset = train_test_split["train"]
eval_dataset = train_test_split["test"]

# Configure LoRA with reduced rank
lora_config = LoraConfig(
    r=4,  # Reduced from 8 to 4 to lower memory usage
    lora_alpha=8,  # Adjusted to 2 * r
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.3,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)

# Training arguments with reduced batch size
training_args = TrainingArguments(
    output_dir="./finetuned_deepseek_lora",
    num_train_epochs=5,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=2,  # Reduced from 4 to 2 to lower memory usage
    learning_rate=2e-5,
    fp16=True,
    logging_steps=10,
    save_steps=50,
    save_total_limit=2,
    remove_unused_columns=False,
    gradient_checkpointing=True,
    warmup_ratio=0.1,
    lr_scheduler_type="cosine",
    evaluation_strategy="steps",
    eval_steps=50,
    load_best_model_at_end=True,
    metric_for_best_model="loss",
)

# Define a data collator
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# Initialize the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=data_collator
)

# Clear memory before training
torch.cuda.empty_cache()
gc.collect()

# Fine-tune the model
trainer.train()

# Merge the LoRA adapters into the base model
model = model.merge_and_unload()

# Save the fine-tuned model (with merged weights)
model.save_pretrained("./finetuned_deepseek_lora")
tokenizer.save_pretrained("./finetuned_deepseek_lora")
