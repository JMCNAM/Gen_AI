import os
import mariadb
import array
import numpy as np
import re
from dotenv import load_dotenv

from langchain.embeddings import HuggingFaceEmbeddings
from langchain_together import Together

load_dotenv()

# Config
DB_CONFIG = {
    "host": "localhost",
    "user": os.getenv("DB_USER"),
    "password": os.getenv("DB_PASSWORD"),
    "database": os.getenv("DB_NAME"),
}

# Load embeddings and LLM
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
llm = Together(model="mistralai/Mistral-7B-Instruct-v0.1", max_tokens=300)

# 1. Get embedding of query
def embed_query(question: str):
    vector = embedding_model.embed_query(question)
    return array.array("f", vector).tobytes()

# 2. Cosine similarity for manual embedding comparison
def cosine_similarity(a, b):
    a = np.array(a)
    b = np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def retrieve_similar_chunks(question: str, k=20):
    conn = mariadb.connect(**DB_CONFIG)
    cursor = conn.cursor()

    query_vector = embedding_model.embed_query(question)

    cursor.execute("SELECT content, embedding FROM documents")
    all_rows = cursor.fetchall()

    similarities = []
    for content, blob in all_rows:
        vec = np.frombuffer(blob, dtype=np.float32)
        score = cosine_similarity(query_vector, vec)
        similarities.append((score, content))

    top_k = sorted(similarities, reverse=True, key=lambda x: x[0])[:k]
    top_chunks = list(dict.fromkeys([item[1] for item in top_k]))

    cursor.close()
    conn.close()
    return top_chunks

# Clean out embedded Q&A structures
def clean_qa_formatting(text):
    return re.sub(r"(?i)question:\s.*?\nanswer:\s.*?(\n|$)", "", text).strip()

def filter_context(context_chunks, user_question):
    user_keywords = set(user_question.lower().split())
    def is_relevant(chunk):
        return any(word in chunk.lower() for word in user_keywords)
    return [chunk for chunk in context_chunks if is_relevant(chunk)]

# 3. Query LLM with cleaned and focused prompt
def ask_question(question: str):
    context_chunks = retrieve_similar_chunks(question, k=50)
    cleaned_chunks = [clean_qa_formatting(chunk) for chunk in context_chunks]
    filtered_chunks = filter_context(cleaned_chunks, question)
    context_text = "\n---\n".join(filtered_chunks)

    prompt = f"""
Answer the following question using only the provided excerpts from a financial document. 
Only respond to the user’s question and nothing else. Return one sentence maximum.

Excerpts:
{context_text}

User question: {question}
Assistant:"""

    response = llm.invoke(prompt, stop=["\nQuestion:", "\nUser question:"])
    return response

# 🔍 Try it
if __name__ == "__main__":
    question = input("Ask a question about the PDF: ")
    answer = ask_question(question)
    print("\n🧠 LLM Answer:\n", answer)
