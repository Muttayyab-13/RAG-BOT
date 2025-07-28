from langchain.llms import LlamaCpp
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

def create_rag_chain(vectorstore):
    # ✅ Use LangChain-compatible LlamaCpp wrapper
    llm = LlamaCpp(
        model_path="./models/mistral-7b-instruct-v0.1.Q4_K_M.gguf",  # Path to your local model
        n_ctx=2048,
        temperature=0.1,
        top_p=0.95,
        max_tokens=512,
        n_threads=6,
        verbose=True
    )

    # Prompt template
    prompt_template = """Use the following context to answer the question. If you don't know the answer, say "I don't know".

Context:
{context}

Question: {question}

Answer:"""

    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )

    # Create RetrievalQA chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
        chain_type_kwargs={"prompt": prompt}
    )

    return qa_chain
