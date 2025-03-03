import streamlit as st
from pathlib import Path
from data_preprocessing import process_docs
from rag import create_rag_chain
import time

def response_generator(prompt,chain):
    response = chain.invoke(prompt)
    for word in response.split():
        yield word + " "
        time.sleep(0.05)



# Set up the file uploader
# uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

# Specify the directory to save files
save_directory = "docs"
save_path="docs/file.pdf"



    

st.title("📝 InsureAgent")
with st.sidebar:
    uploaded_file = st.file_uploader("Upload a document", type=("pdf"))
    if uploaded_file is not None:
        with open(save_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.success(f"File saved successfully: {save_path}")
    retriever=process_docs(save_path)
    chain,chain_with_sources=create_rag_chain(retriever)

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
# Streamed response emulator

if prompt := st.chat_input("What is up?"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        response = st.write_stream(response_generator(prompt,chain))
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})







