import streamlit as st
from main import create_qa_engine
qa = create_qa_engine()

# Streamlit UI
# ===============
st.set_page_config(page_title="Anwer your Question in Biology!", page_icon=":robot:")
st.header("Query PDF Source")

form_input = st.text_input('Enter Query')
submit = st.button("Generate")

if submit:
    result = qa.invoke({'query': form_input})
    print("Answer: ",  result['result'])
    st.write(result['result'])