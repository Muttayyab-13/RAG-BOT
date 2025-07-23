from langchain.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
from transformers import pipeline
import os
from huggingface_hub import login

load_dotenv()
token = os.getenv("Hugging_face_token")
login(token)

def create_rag_chain(vectorstore):
    llm_pipeline = pipeline(
        model="mistralai/Mistral-7B-Instruct-v0.1",
        task="text-generation",
        tokenizer="mistralai/Mistral-7B-Instruct-v0.1",
        temperature=0.1,
        token=token,
        max_new_tokens=512  # moved here from model_kwargs
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
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
        chain_type_kwargs={"prompt": PROMPT}
    )

    return qa_chain
