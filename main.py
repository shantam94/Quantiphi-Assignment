from utils.data_read import *
from utils.model import *

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
        print("-"*100)


data = load_pdf_document(file_path = 'data\ConceptsofBiology-WEB.pdf')
docs = split_data(data)

embeddings = create_embeddings(device = 'cpu', modelPath ="sentence-transformers/all-MiniLM-L6-v2" )

db = create_vector_store(docs, embeddings)

llm = create_model(model_name = "google/flan-t5-base", model_temp = 0.1)
qa = create_retreiver(db, llm, top_docs_to_search = 4)

# questions = ["what is Biology?",
#              "What are the properties of life? List them",
#              "What is chemotaxis?",
#              "What is adaptation? ",
#              "What does genes provide? Explain",
#              "What is an atom?",
#              "What is an organ system?"]
# for el in questions[:1]:
#     result = qa.invoke({'query': el})
#     print("Query: ",result['query'])
#     print("Answer: ",  result['result'])
#     print("-"*100)

get_user_query(qa)
