import os
from langchain.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter


def load_documents(folder_path):

    documents = []
    
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path,filename)
        if filename.endswith('.pdf'):
            loader=PyPDFLoader(file_path)
        elif filename.endswith('.txt'):
            loader=TextLoader(file_path)
        else:
            continue
        documents.extend(loader.load())

    return documents


def split_documents(documents):
    text_splitter=RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    return text_splitter.split_documents(documents)


