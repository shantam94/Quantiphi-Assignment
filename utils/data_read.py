from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from .logging_decorator import log_function_call
from PyPDF2 import PdfReader
import chromadb
import os
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter


@log_function_call
def load_pdf_document(file_path):
    loader = PyPDFLoader(file_path)
    data = loader.load()
    # Pages 19 to 76 are the two chapters
    data = data[18:36]
    print("-"*20, "Data Read Complete ...", "-"*20)
    return data


@log_function_call
def split_data(data):
    # Create an instance of the RecursiveCharacterTextSplitter class with specific parameters.
    # It splits text into chunks of 1000 characters each with a 150-character overlap.
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)

    # 'data' holds the text you want to split, split the text into documents using the text splitter.
    docs = text_splitter.split_documents(data)
    print("-"*20, "Data Split Complete ...", "-"*20)
    return docs

@log_function_call
def load_chunk_persist_pdf(modelPath = "sentence-transformers/all-MiniLM-L6-v2", model_kwargs = {'device':'cpu'},\
    encode_kwargs = {'normalize_embeddings': False}) -> Chroma:
    pdf_folder_path = "data"
    documents = []
    for file in os.listdir(pdf_folder_path):
        if file.endswith('.pdf'):
            pdf_path = os.path.join(pdf_folder_path, file)
            loader = PyPDFLoader(pdf_path)
            documents.extend(loader.load())
    documents = documents[16:76]
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunked_documents = text_splitter.split_documents(documents)
    client = chromadb.Client()
    if client.list_collections():
        consent_collection = client.create_collection("biology_book")
    else:
        print("Collection already exists")
    vectordb = Chroma.from_documents(
        documents=chunked_documents,
        embedding=HuggingFaceEmbeddings(
    model_name=modelPath,     # Provide the pre-trained model's path
    model_kwargs=model_kwargs, # Pass the model configuration options
    encode_kwargs=encode_kwargs # Pass the encoding options
),
        persist_directory="chroma_store"
    )
    vectordb.persist()
    return vectordb