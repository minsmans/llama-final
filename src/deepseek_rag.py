import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from langchain_community.embeddings import HuggingFaceEmbeddings  # Updated import
from langchain.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain.docstore.document import Document

# Load fine-tuned model
model_path = "./deepseek_lora_output"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Setup RAG
with open("disaster_docs.txt", "r") as f:
    raw_text = f.read()
splitter = CharacterTextSplitter(chunk_size=200, chunk_overlap=20)
docs = [Document(page_content=chunk) for chunk in splitter.split_text(raw_text)]
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vector_store = FAISS.from_documents(docs, embeddings)

# Process with RAG
def process_disaster_rag(data_feed):
    retrieved_docs = vector_store.similarity_search(data_feed, k=1)
    context = retrieved_docs[0].page_content if retrieved_docs else "No context available."
    
    prompts = [
        f"Analyze seismic data: {data_feed}\nContext: {context}",
        f"Mitigate hurricane: {data_feed}\nContext: {context}",
        f"Recover from wildfire: {data_feed}\nContext: {context}"
    ]
    responses = {}
    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        outputs = model.generate(**inputs, max_length=100)
        responses[prompt.split(":")[0]] = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return responses

# Sample feeds
data_feeds = [
    "5.0 magnitude near Seattle",
    "Tropical storm 60mph near Houston",
    "Wildfire smoke over Colorado"
]

# Process
for feed in data_feeds:
    result = process_disaster_rag(feed)
    print(f"Data Feed: {feed}")
    for action, response in result.items():
        print(f"{action}: {response}")
    print("---")

# Integration
def integrate_with_interlinked(result):
    print("Sending to Interlinked system:", result)

integrate_with_interlinked(result)