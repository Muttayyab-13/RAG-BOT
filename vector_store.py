from langchain_community.vectorstores.chroma import Chroma
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from dotenv import load_dotenv
import os

load_dotenv()

embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
PERSIST_DIR = "./chroma_db"

def create_vector_store(documents):
    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embedding_model,  # 👈 USE "embedding", not "embedding_function"
        persist_directory=PERSIST_DIR,
    )
    vectorstore.persist()
    return vectorstore


def load_vector_store():
    vectorstore = Chroma(
        persist_directory=PERSIST_DIR,
        embedding=embedding_model,  # 👈 USE "embedding", not "embedding_function"
    )
    return vectorstore
