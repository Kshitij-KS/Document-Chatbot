from langchain_community.document_loaders import DirectoryLoader, PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv
import os
import shutil

# Constants
CHROMA_PATH = "chroma"
DATA_PATH = "data"

def get_embedding_function():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'}
    )
    return embeddings

def load_documents():
    if not os.path.exists(DATA_PATH):
        os.makedirs(DATA_PATH, exist_ok=True)
        print(f"Created directory: {DATA_PATH}")
        print("Please add your .md or .pdf files to this directory and run again.")
        return []
    
    md_loader = DirectoryLoader(DATA_PATH, glob="*.md")
    md_documents = md_loader.load()

    pdf_loader = PyPDFDirectoryLoader(DATA_PATH)
    pdf_documents = pdf_loader.load()

    documents = md_documents + pdf_documents
    print(f"Loaded {len(documents)} documents ({len(md_documents)} markdown, {len(pdf_documents)} PDF) from {DATA_PATH}")
    return documents

def split_text(documents: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=100,
        length_function=len,
        add_start_index=True,
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Split {len(documents)} documents into {len(chunks)} chunks.")

    if len(chunks) > 10:
        document = chunks[10]
        print(document.page_content)
        print(document.metadata)
    elif len(chunks) > 0:
        document = chunks[0]
        print(document.page_content)
        print(document.metadata)
    else:
        print("No chunks were created.")

    return chunks

def save_to_chroma(chunks: list[Document]):
    if not chunks:
        print("No chunks to save.")
        return
    
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)

    db = Chroma.from_documents(
        chunks, 
        get_embedding_function(), 
        persist_directory=CHROMA_PATH
    )
    print(f"Saved {len(chunks)} chunks to {CHROMA_PATH}.")

def generate_data_store():
    documents = load_documents()
    if not documents:
        return
    chunks = split_text(documents)
    save_to_chroma(chunks)

def main():
    generate_data_store()

if __name__ == "__main__":
    main()
