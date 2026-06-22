# 📄 Retrieval-Augmented Generation (RAG) Chatbot using LangChain and OpenAI

## 🚀 Overview

This project is a **Retrieval-Augmented Generation (RAG) chatbot** built using Python, LangChain, and OpenAI.

It allows users to:
- Upload PDF documents
- Ask questions based on document content
- Get AI-generated context-aware answers

The system combines **information retrieval + Large Language Models (LLMs)** to deliver accurate and intelligent responses.

---

## 🧠 What is RAG?

Retrieval-Augmented Generation (RAG) enhances Large Language Models by retrieving relevant information from external documents before generating responses.

Instead of relying only on training data, the model:
1. Retrieves relevant document chunks
2. Passes them as context to the LLM
3. Generates accurate answers

---

## 🏗️ Architecture

User Upload → PDF Loader → Text Splitter → Embeddings → Vector DB (ChromaDB) → Retriever → OpenAI LLM → Response

---

## 🛠️ Tech Stack

- Python
- LangChain
- OpenAI API (GPT-4o-mini)
- ChromaDB (Vector Database)
- Streamlit (UI)
- PyPDF

---

## 📂 Project Structure

rag-chatbot/
│
├── app.py
├── rag_pipeline.py
├── requirements.txt
├── .env
├── README.md
└── data/
└── sample.pdf


---

## ⚙️ Installation & Setup

### 1. Clone the repository
```bash
git clone https://github.com/your-username/rag-chatbot-langchain-openai.git
cd rag-chatbot-langchain-openai

pip install -r requirements.txt

OPENAI_API_KEY=your_openai_api_key_here

streamlit run app.py
