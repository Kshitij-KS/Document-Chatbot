# Document Chatbot

A Python-based document chatbot that uses vector embeddings and retrieval-augmented generation (RAG) to answer questions about your documents. Built with LangChain, HuggingFace embeddings, and Chroma vector database.

## Features

- **Document Processing**: Load and process Markdown and PDF documents
- **Vector Embeddings**: Uses HuggingFace sentence-transformers for free, high-quality embeddings
- **Semantic Search**: Find relevant document chunks using similarity search
- **Question Answering**: Generate contextual answers based on retrieved document content
- **Local Processing**: No API costs for embeddings (HuggingFace models run locally)
- **Flexible LLM Support**: Compatible with Groq API for free and fast generation

## Project Structure

Document Chatbot/
├── createDatabase.py # Document processing and vector database creation
├── query.py # Query interface for asking questions
├── requirements.txt # Python dependencies
├── .env # Environment variables (API keys)
├── .gitignore # Git ignore rules
├── data/ # Document storage directory
│ └── *.md # Markdown documents
└── chroma/ # Vector database (auto-generated)


## Installation

### 1. Clone and Setup

git clone <your-repo-url>
cd "Document Chatbot"
python -m venv venv
source venv/bin/activate # On Windows: venv\Scripts\activate


### 2. Install Dependencies

pip install -r requirements.txt
pip install "unstructured[md]"


### 3. Environment Setup

Create a `.env` file in the project root:

GROQ_API_KEY=your_groq_api_key_here

## Usage

### Step 1: Prepare Your Documents

Place your documents in the `data/` directory:
- Supported formats: Markdown (`.md`), PDF (`.pdf`)
- The system will automatically create the directory if it doesn't exist

### Step 2: Create Vector Database

Process your documents and create the vector database:

python createDatabase.py


### Step 3: Query Your Documents

Ask questions about your documents:

python query.py "What is the main character's name?"
python query.py "How does Alice meet the Mad Hatter?" --threshold 0.4
