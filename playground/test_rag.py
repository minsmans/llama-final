from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from langchain_community.embeddings import HuggingFaceEmbeddings
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

# Test RAG
data_feed = "5.0 magnitude near Seattle"
retrieved_docs = vector_store.similarity_search(data_feed, k=1)
context = retrieved_docs[0].page_content if retrieved_docs else "No context available."
prompt = f"Analyze seismic data: {data_feed}\nContext: {context}"
inputs = tokenizer(prompt, return_tensors="pt").to(device)
outputs = model.generate(**inputs, max_new_tokens=50)  # Updated to max_new_tokens
print(f"Data Feed: {data_feed}")
print(f"Context: {context}")
print(f"Response: {tokenizer.decode(outputs[0], skip_special_tokens=True)}")