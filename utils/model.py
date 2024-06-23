from langchain_huggingface import HuggingFaceEmbeddings
# from langchain.vectorstores import FAISS
from transformers import AutoTokenizer, pipeline
# from langchain import HuggingFacePipeline
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from .logging_decorator import log_function_call



@log_function_call
def create_embeddings(device = 'cpu', modelPath ="sentence-transformers/all-MiniLM-L6-v2" ):
    
    # Define the path to the pre-trained model you want to use
    modelPath = modelPath
    # modelPath = "google/flan-t5-large"

    # Create a dictionary with model configuration options, specifying to use the CPU for computations
    model_kwargs = {'device':device}

    # Create a dictionary with encoding options, specifically setting 'normalize_embeddings' to False
    encode_kwargs = {'normalize_embeddings': False}

    # Initialize an instance of HuggingFaceEmbeddings with the specified parameters
    embeddings = HuggingFaceEmbeddings(
        model_name=modelPath,     # Provide the pre-trained model's path
        model_kwargs=model_kwargs, # Pass the model configuration options
        encode_kwargs=encode_kwargs) # Pass the encoding options
    print("-"*20, "Embeddings Created ...", "-"*20)
    return embeddings

@log_function_call
def create_vector_store(docs, embeddings):
    db = FAISS.from_documents(docs, embeddings)
    print("-"*20, "VEctor Store Created ...", "-"*20)
    return db

@log_function_call
def create_model(model_name = "google/flan-t5-base", model_temp = 0.1):
    
    # Specify the model name you want to use
    model_name = model_name

    # Load the tokenizer associated with the specified model
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding=True, truncation=True, max_length=512)

    # Define a question-answering pipeline using the model and tokenizer
    question_answerer = pipeline(
        "text2text-generation",
        model=model_name,
        tokenizer=tokenizer,
        max_length=512
        # ,return_tensors='pt'
    )

    # Create an instance of the HuggingFacePipeline, which wraps the question-answering pipeline
    # with additional model-specific arguments (temperature and max_length)
    llm = HuggingFacePipeline(
        pipeline=question_answerer,
        model_kwargs={"temperature": model_temp, "max_length": 512}
    )
    print("-"*20, "Model Loaded Created ...", "-"*20)
    return llm


@log_function_call
def create_retreiver(db, llm, top_docs_to_search = 4):
    # Create a retriever object from the 'db' with a search configuration where it retrieves up to 4 relevant splits/documents.
    retriever = db.as_retriever(search_kwargs={"k": top_docs_to_search})

    # Create a question-answering instance (qa) using the RetrievalQA class.
    # It's configured with a language model (llm), a chain type "refine," the retriever we created, and an option to not return source documents.
    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever, return_source_documents=True)
    
    print("-"*20, "Retreiver Created ...", "-"*20)
    
    return qa