from vector_store import load_vector_store
from rag_chain import create_rag_chain

def main():
    # Load vector store

    vectorestore=load_vector_store()

    # Create rag chain

    qa_chain=create_rag_chain(vectorstore)

    print("RAG Bot is ready ! Type 'quit' to exit")

    while True:
        question = input("\nAsk a question: ")
        if question.lower() == 'quit':
            break
        
        try:
            response = qa_chain.run(question)
            print(f"Answer: {response}")
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    main()