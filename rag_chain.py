from langchain.llms import HuggingFacePipeline
from langchain.chains import retrieval_qa
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
from transformers import pipeline
import os

token=os.getenv("Hugging_face_token")

def create_rag_chain(vectorstore):
    llm_pipeline = pipeline(
        model="mistralai/Mistral-7B-Instruct-v0.1",
        task="text-generation",
        model_kwargs={"temperature": 0.1, "max_new_tokens": 512},
        use_auth_token=token  
    )

    llm = HuggingFacePipeline(pipeline=llm_pipeline)

    # PROMPT
    prompt_template = """Use the following context to answer the question at the end. If you don't know the answer, say "I don't know".

Context:
{context}

Question: {question}

Answer:"""

    PROMPT = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )

    # Build Retrieval QA
    qa_chain = retrieval_qa.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
        chain_type_kwargs={"prompt": PROMPT}
    )

    return qa_chain
