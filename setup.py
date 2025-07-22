from document_loader import load_documents, split_documents
from vector_store import create_vector_store

def setup_rag_system():
    print("Loading documents...")
    documents = load_documents("documents")
    
    print("Splitting documents...")
    split_docs = split_documents(documents)
    
    print("Creating vector store...")
    vectorstore = create_vector_store(split_docs)
    
    print("Setup complete!")

if __name__ == "__main__":
    setup_rag_system()