from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from .logging_decorator import log_function_call

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
