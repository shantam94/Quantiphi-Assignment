from utils.data_read import *
from utils.model import *
import streamlit as st
import argparse
import warnings
# Suppress all warnings
warnings.simplefilter("ignore")



@log_function_call
def get_user_query(qa):
    while True:
        user_input = input("Enter something (type 'end' to stop): ")
        if user_input.lower() == 'end':
            break
        print(f"Query: {user_input}")
        result = qa.invoke({'query': user_input})
        print("Answer: ",  result['result'])
        print("Page Reference: ",  result['result'])
        print("-"*100)


# def create_qa_engine():

#     vectordb = load_chunk_persist_pdf()
#     llm = create_model(model_name = "google/flan-t5-base", model_temp = 0.5)
#     retriever = vectordb.as_retriever(search_kwargs={"k": 1})
#     qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever, return_source_documents=True)
#     return qa



parser = argparse.ArgumentParser(description='Biology QnA')
parser.add_argument('--db', type=str, help='database type , f for FAISS and c for chromadb', default='f')

args = parser.parse_args()


@st.cache_data
def model_main():
    if args.db == 'f':
        print("FAISS database selected! ")
        
        #  Load data in a dataloader
        data = load_pdf_document(file_path = r'data/ConceptsofBiology-WEB.pdf')
        
        # Split data and do chunking
        docs = split_data(data)
        # create embeddings
        embeddings = create_embeddings(device = 'cpu', modelPath ="sentence-transformers/all-MiniLM-L6-v2" )
        # create vector database
        db = create_vector_store(docs, embeddings)
        # load llm
        llm = create_model(model_name = "google/flan-t5-base", model_temp = 0.5)
        # create a retreiver
        retriever = db.as_retriever(search_kwargs={"k": 1})
        qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever, return_source_documents=True)
    else:
        vectordb = load_chunk_persist_pdf()
        llm = create_model(model_name = "google/flan-t5-base", model_temp = 0.5)
        retriever = vectordb.as_retriever(search_kwargs={"k": 1})
        qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever, return_source_documents=True)
    return qa


def main():

    # Streamlit UI
    # ===============
    st.set_page_config(page_title="Anwer your Question in Biology!", page_icon=":robot:")
    
    qa = model_main()
    
    st.header("Query PDF Source",divider='rainbow')

    questions = ["what is Biology?",
             "What are the properties of life? List them",
             "What is chemotaxis?",
             "What is adaptation? ",
             "What does genes provide? Explain",
             "What is an atom?",
             "What is an organ system?"]
    
    st.subheader("Example Questions")
    for question in questions:
            st.markdown(f"- {question}")
    
    form_input = st.text_input('Enter Query')
    submit = st.button("Answer")

    if submit:
        result = qa.invoke({'query': form_input})
        print("Answer: ",  result['result'])
        st.write(result['result'])
    
    
    
if __name__ == '__main__':
    main()