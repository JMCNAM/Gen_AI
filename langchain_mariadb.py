import os
import array
import mariadb
from dotenv import load_dotenv
from langchain.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings

# Load environment variables
load_dotenv()

# Database connection settings
DB_CONFIG = {
    "host": "localhost",
    "user": os.getenv("DB_USER"),
    "password": os.getenv("DB_PASSWORD"),
    "database": os.getenv("DB_NAME"),
}

# Step 1: Load and chunk the PDF
pdf_path = "C:/Users/james/Downloads/Netflix-10-K-01262024.pdf"  # Replace with your actual PDF
loader = PyMuPDFLoader(pdf_path)
documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = text_splitter.split_documents(documents)

# Step 2: Generate embeddings
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
embeddings = embedding_model.embed_documents([chunk.page_content for chunk in chunks])

# Step 3: Connect to MariaDB
conn = mariadb.connect(**DB_CONFIG)
cursor = conn.cursor()

# Step 4: Create table if not exists
cursor.execute("""
CREATE TABLE IF NOT EXISTS documents (
    id INT AUTO_INCREMENT PRIMARY KEY,
    content TEXT NOT NULL,
    embedding VECTOR(384)  -- match HuggingFace embedding size
)
""")

# Step 5: Insert each chunk
for chunk, vector in zip(chunks, embeddings):
    binary_vector = array.array("f", vector).tobytes()
    cursor.execute(
        "INSERT INTO documents (content, embedding) VALUES (?, ?)",
        (chunk.page_content, binary_vector)
    )
conn.commit()
cursor.close()
conn.close()

print("✅ PDF uploaded and embedded to MariaDB Vector DB.")
