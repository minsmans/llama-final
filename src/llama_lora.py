import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TrainingArguments, DataCollatorForLanguageModeling, Trainer
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model
from datasets import Dataset, load_dataset

print("--- Starting Llama LoRA Script ---")

# 1. 모델 ID 설정
model_id = "meta-llama/Llama-3.1-8B-Instruct"
print(f"--- Model ID: {model_id} ---")

# 2. 4비트 양자화를 위한 BitsAndBytes 설정
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

# 3. 토크나이저 및 모델 불러오기
print("--- Loading tokenizer and model (this may take a while)... ---")
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True
)
print("--- Model and tokenizer loaded ---")

# 4. k-bit 훈련을 위해 모델 준비
model.gradient_checkpointing_enable()
model = prepare_model_for_kbit_training(model)

# 5. LoRA 설정
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# 6. 데이터셋 로딩 및 전처리 (샘플 데이터셋으로 변경)
print("--- Loading a sample dataset from Hugging Face Hub to verify the pipeline ---")
raw_dataset = load_dataset("Abirate/english_quotes", split="train")

def adapt_sample_dataset(example):
    return {
        "instruction": example["quote"],
        "output": example["quote"]
    }
raw_dataset = raw_dataset.map(adapt_sample_dataset)


def formatting_prompts_func(examples):
    instructions = examples['instruction']
    outputs = examples['output']
    texts = []
    for instruction, output in zip(instructions, outputs):
        messages = [
            {"role": "user", "content": instruction},
            {"role": "assistant", "content": output}
        ]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        texts.append(text)
    return {"text": texts}

dataset = raw_dataset.map(formatting_prompts_func, batched=True)

def tokenize_function(examples):
    tokenized_output = tokenizer(examples["text"], truncation=True, max_length=512)
    tokenized_output["labels"] = tokenized_output["input_ids"].copy()
    return tokenized_output

tokenized_dataset = dataset.map(tokenize_function, batched=True)

# *** 최종 수정: 훈련에 필요한 컬럼만 남기고 모두 제거 ***
columns_to_keep = ['input_ids', 'attention_mask', 'labels']
columns_to_remove = [col for col in tokenized_dataset.column_names if col not in columns_to_keep]
tokenized_dataset = tokenized_dataset.remove_columns(columns_to_remove)

print(f"--- Data processing complete. Final columns: {tokenized_dataset.column_names} ---")

# 7. 훈련 인자(Argument) 설정
output_path = "/content/deepseek-playground/llama_lora_output_SAMPLE"
training_args = TrainingArguments(
    output_dir=output_path,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    num_train_epochs=1,
    logging_steps=1,
    save_total_limit=2,
    fp16=True,
    remove_unused_columns=False,
)

# 기본 'Trainer'를 사용
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    tokenizer=tokenizer,
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
)

# 훈련 시작
print("\\n--- Starting training ---")
trainer.train()
print("--- 훈련이 완료되었습니다! ---")