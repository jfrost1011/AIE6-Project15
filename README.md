---
title: Open-Source RAG Demo – LangChain × Hugging Face × Chainlit
emoji: 🔍
colorFrom: pink
colorTo: indigo
sdk: docker
sdk_version: "1.0"
app_file: app.py
pinned: false
---

# Open-Source RAG Demo  
*LLM & embedding endpoints on Hugging Face | LangChain 0.2 | Chainlit UI*

<div align="center">
  <img src="https://github.com/jfrost1011/AIE6-Project15/assets/paul-graham-bot-banner.png" width="85%" />
</div>

A compact **Retrieval-Augmented Generation** project built during the *AI Maker Space – Session 15* lab:

1. **Deploy** open-source endpoints (LLM + Embeddings) to **Hugging Face Inference Endpoints**  
2. **Prototype** a LangChain v0.2 pipeline in a Jupyter notebook (Cursor IDE)  
3. **Serve** the pipeline with a **Chainlit** chat UI – runnable locally *and* as a HF Space (Docker)  

The demo answers questions about Paul Graham’s essays using a FAISS vector store and a Llama-3 8 B Instruct model.

---

## ✨ Key pieces

| Component | Tech / Model | Purpose |
|-----------|--------------|---------|
| **LLM endpoint** | `NousResearch/Meta-Llama-3-8B-Instruct` (HF Inference Endpoint, L4 GPU) | generates answers |
| **Embedding endpoint** | `BAAI/bge-base-en-v1.5` (HF Inference Endpoint, CPU) | text → vectors |
| **Vector store** | `langchain_community.vectorstores.FAISS` | similarity search |
| **RAG chain** | LangChain LCEL → `Retriever ➜ Prompt ➜ HuggingFaceEndpoint` | orchestration |
| **UI** | Chainlit 2 | chat front-end |
| **Container** | Docker Space (Python 3.13, uv) | HF deployment |

---

## 🗂 Project layout

open-source-rag-chainlit/
├─ app.py # Chainlit app (LCEL RAG chain)
├─ Dockerfile # Space image
├─ requirements.txt # runtime deps
├─ data/
│ └─ paul_graham_essays.txt
├─ chainlit.md # welcome screen (Markdown)
└─ README.md # ← you are here


---

## 🚀 Quick start

### 1 · Run locally

```bash
git clone https://github.com/jfrost1011/open-source-rag-chainlit.git
cd open-source-rag-chainlit

# create venv & install deps
uv venv && uv pip install -r requirements.txt

# copy your credentials
cp .env.sample .env           # edit with your HF_… endpoints + token

# launch
chainlit run app.py

Open http://localhost:8000.

Run on Hugging Face
The repo is mirrored to https://huggingface.co/spaces/jfrost10/open-source-rag-chainlit.
