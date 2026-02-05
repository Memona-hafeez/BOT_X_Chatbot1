# Ollama RAG Chatbot (Document-Based AI Assistant)

A document-based **Retrieval-Augmented Generation (RAG)** chatbot built using **Ollama (local LLMs)** and **ChromaDB** for semantic search.

This system allows users to upload **PDF documents**, store them as vector embeddings, and ask intelligent questions that are answered using **retrieved context**, not hallucinations.

---

## Overview

Traditional chatbots answer questions without understanding your private data.  
This project solves that problem by combining:

- Local LLMs via **Ollama**
- Vector similarity search using **ChromaDB**
- Document-based context retrieval

The result is a **private, local, document-aware AI chatbot**.

---

## Key Features

- Upload and process PDF documents
- Text extraction and chunking
- Embedding generation
- Semantic search using ChromaDB
- Local LLM inference via Ollama
- Context-aware question answering (RAG)
- Flask-based web application

---

## Tech Stack

- **Python**
- **Ollama** (local LLM inference)
- **ChromaDB** (vector database)
- **LangChain / LlamaIndex** (RAG pipeline)
- **Flask** (backend server)
- **PDF Loader** (document parsing)

---

## System Architecture

```
PDF Documents
      ↓
Text Extraction
      ↓
Text Chunking
      ↓
Embedding Generation
      ↓
ChromaDB (Vector Store)
      ↓
Similarity Search
      ↓
Ollama LLM
      ↓
Answer
```

---

## Project Structure

```
.
├── pdfFiles/                    # Uploaded PDF documents
├── chromadb/                    # ChromaDB vector storage
├── llama2VectorDB/              # Vector indexes (alternative backend)
├── nomicDB/                     # Embedding database
├── app.py                       # Flask application
├── loader.py                    # PDF loading and processing
├── requirements.txt
├── README.md
└── venv/
```

---

## How It Works

1. User uploads one or more PDF files  
2. Text is extracted from documents  
3. Text is split into manageable chunks  
4. Chunks are converted into embeddings  
5. Embeddings are stored in **ChromaDB**  
6. User submits a question  
7. Relevant chunks are retrieved via similarity search  
8. Retrieved context is passed to **Ollama LLM**  
9. The LLM generates an accurate, grounded answer  

This ensures responses are based strictly on uploaded documents.

---

## Ollama Integration

This project uses **Ollama** to run large language models locally.

Benefits of using Ollama:
- No cloud dependency
- Full data privacy
- Fast local inference
- Support for models like LLaMA, Mistral, etc.

Make sure Ollama is running before starting the app.

---

## ChromaDB Functionality

**ChromaDB** is used as the vector database to:

- Store document embeddings
- Perform similarity search
- Retrieve relevant context for user queries
- Enable fast and scalable semantic retrieval

It is a core component of the RAG pipeline.

---

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/Memona-hafeez/BOT_X_Chatbot1.git
cd BOT_X_Chatbot1
```

### 2. Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate   # Linux / Mac
venv\Scripts\activate      # Windows
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Ollama Setup

Install Ollama and pull a model:

```bash
ollama pull llama2
```

Start Ollama service before running the app.

---

## Run the Application

```bash
python app.py
```

Open your browser and visit:

```
http://localhost:5000
```

---

## Example Usage

1. Launch the web interface  
2. Upload PDF documents  
3. Ask questions related to document content  
4. Receive AI-generated answers grounded in your data  

---

## Limitations

- Designed for PDFs only
- No authentication system
- Large PDFs may increase processing time
- Requires local system resources for Ollama models

---

## Future Improvements

- Support for DOCX and TXT files
- UI enhancement with modern frontend
- Persistent multi-user sessions
- Advanced document metadata filtering
- Dockerized deployment

---

## License

This project is open source.  
Add a LICENSE file (MIT or Apache 2.0 recommended) before production deployment.
