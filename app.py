import streamlit as st
from vector_store import load_vector_store
from rag_chain import create_rag_chain

@st.cache_resource
def load_rag_system():
    vectorstore = load_vector_store()
    qa_chain = create_rag_chain(vectorstore)
    return qa_chain

def main():
    st.title("RAG Bot")
    
    qa_chain = load_rag_system()
    
    question = st.text_input("Ask a question:")
    
    if st.button("Get Answer"):
        if question:
            with st.spinner("Thinking..."):
                response = qa_chain.run(question)
                st.write("Answer:", response)

if __name__ == "__main__":
    main()