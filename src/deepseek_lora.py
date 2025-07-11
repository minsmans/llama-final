# deepseek_lora.py
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model
from datasets import load_dataset

# Load model and tokenizer
model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Configure LoRA
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, lora_config)

# Load dataset
dataset = load_dataset("json", data_files="disaster_data.jsonl")
print(f"Dataset size: {len(dataset['train'])} examples")

# Tokenize with consistent max_length
def tokenize_function(examples):
    instructions = examples["instruction"]
    outputs = examples["output"]
    # Combine instruction and output with a separator
    combined = [f"{instr}\n\nAnalysis:\n{out}" for instr, out in zip(instructions, outputs)]
    encodings = tokenizer(
        combined,
        truncation=True,
        padding="max_length",
        max_length=768,  # Increased to fit both input and output
        return_tensors="pt"
    )
    input_ids = encodings["input_ids"]
    attention_mask = encodings["attention_mask"]
    labels = input_ids.clone()
    # Mask input portion in labels
    for i in range(len(instructions)):
        input_len = len(tokenizer.encode(instructions[i], add_special_tokens=False)) + len(tokenizer.encode("\n\nAnalysis:\n", add_special_tokens=False))
        labels[i, :input_len] = -100
    print(f"Input IDs shape: {input_ids.shape}, Labels shape: {labels.shape}")
    return {
        "input_ids": input_ids.squeeze(),
        "attention_mask": attention_mask.squeeze(),
        "labels": labels.squeeze()
    }

tokenized_dataset = dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=["instruction", "output"]
)

# Training setup
training_args = TrainingArguments(
    output_dir="./deepseek_lora_output_training_args",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    num_train_epochs=5,
    learning_rate=2e-5,
    fp16=True,
    logging_steps=1,
    save_steps=5,
    save_total_limit=2,
    remove_unused_columns=False,
    optim="adamw_torch"
)

class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)
        labels = inputs["labels"].to(device)
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        print(f"Step loss: {loss.item()}")
        return (loss, outputs) if return_outputs else loss

trainer = CustomTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"]
)

total_samples = len(tokenized_dataset["train"])
effective_batch_size = training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps
total_steps = (total_samples // effective_batch_size) * training_args.num_train_epochs
print(f"Total training steps: {total_steps}")  # 50 with 10 samples, 5 epochs

# Fine-tune
trainer.train()

# Save
model.save_pretrained("./deepseek_lora_output")
tokenizer.save_pretrained("./deepseek_lora_output")