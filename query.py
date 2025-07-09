import argparse
from langchain_chroma import Chroma  # Updated import
from langchain_huggingface import HuggingFaceEmbeddings  # Match database embeddings
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from dotenv import load_dotenv
import os

# Loading env variables
load_dotenv()

CHROMA_PATH = "chroma"

PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""

def get_embedding_function():
    # Same embed function as used in createDatabase.py
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'}
    )
    return embeddings

def main():
    # Creating CLI
    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str, help="The query text.")
    args = parser.parse_args()
    query_text = args.query_text

    embedding_function = get_embedding_function()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    # Searching the DB
    results = db.similarity_search_with_relevance_scores(query_text, k=3)
    
    # Debug: Printing the search results
    print(f"Found {len(results)} results")
    for i, (doc, score) in enumerate(results):
        print(f"Result {i+1}: Score = {score:.3f}")
        print(f"Content preview: {doc.page_content[:100]}...")
    
    if len(results) == 0 or results[0][1] < 0.4:
        print(f"Unable to find matching results.")
        return


    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)
    print(prompt)

    try:
        model = ChatOpenAI()
        response = model.invoke(prompt) 
        response_text = response.content
    except Exception as e:
        print(f"Error with OpenAI API: {e}")
        return

    sources = [doc.metadata.get("source", None) for doc, _score in results]
    formatted_response = f"Response: {response_text}\nSources: {sources}"
    print(formatted_response)

if __name__ == "__main__":
    main()
