# Disaster Prediction and Response System 

## Overview
This project enhances `DeepSeek-R1-Distill-Qwen-1.5B` (1.5 billion parameters) using **LoRA (Low-Rank Adaptation)** for efficient fine-tuning and **RAG (Retrieval-Augmented Generation)** for context-aware responses.

---

## Techniques
### Few-Shot Learning
- **Purpose**: Use the pre-trained `DeepSeek-R1-Distill-Qwen-1.5B` with in-context examples instead of fine-tuning, due to small dataset size (6 samples).
- **Script**: `playground/deepseek_fewshot.py`.
- **Method**: Loads `disaster_data.jsonl` as examples, constructs a prompt, and generates analysis without training.
- **Usage**:
  ```bash
  python deepseek_fewshot.py
### LoRA (Low-Rank Adaptation)
- **Purpose**: Fine-tune the LLM efficiently without updating all 1.5B parameters.
- **Method**: Adds small, trainable adapter matrices to attention layers (`q_proj`, `v_proj`), freezing original weights. Only ~3M parameters are trained (~1-2% of total).
- **Implementation**:
  - Script: `src/deepseek_lora.py`.
  - Dataset: `disaster_data.jsonl` (e.g., "Analyze seismic data: 4.5 magnitude near Cascadia" → "Moderate risk; monitor USGS").
  - Config: `r=16`, `lora_alpha=32`, 3 epochs, batch size 1, `fp16`.
- **Outcome**: Model adapts to disaster reasoning, saved to `./fine_tuned_deepseek_qwen/`.

### RAG (Retrieval-Augmented Generation)
- **Purpose**: Augment the LLM with real-time disaster data for precise responses.
- **Method**: 
  - **Retriever**: Indexes `disaster_docs.txt` (e.g., "USGS: 4.5 magnitude quakes near Cascadia…") using `HuggingFaceEmbeddings` (`all-MiniLM-L6-v2`) and `FAISS`.
  - **Generator**: Fine-tuned Qwen-1.5B generates responses with retrieved context.
- **Why**: Enhances accuracy with external data (e.g., USGS/NOAA/NASA-like reports).
- **Implementation**:
  - Script: `src/deepseek_rag.py`.
  - Process: Retrieves context for feeds (e.g., "5.0 magnitude near Seattle"), generates insights.
- **Outcome**: Context-aware outputs (e.g., "Moderate risk; monitor USGS").

---

## Setup

### Requirements
- **Hardware**: Nvidia GPU
- **Environment**: Conda `deepseek_env` (Python 3.10).
- **Dependencies**:
  ```bash
  conda create -n deepseek_env python=3.10 -y
  conda activate deepseek_env
  conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
  pip install transformers peft datasets langchain-community sentence-transformers faiss-cpu
### Testing:
   - Run test files to validate the setup:
     ```bash
     python test_lora.py
     python test_rag.py
