import streamlit as st
from pathlib import Path
from data_preprocessing import process_docs
from rag import create_rag_chain
import time
import pandas as pd
import os
from datetime import datetime

# Feedback storage setup
FEEDBACK_FILE = "feedback.csv"
if not os.path.exists(FEEDBACK_FILE):
    pd.DataFrame(columns=["timestamp", "query", "response", "rating"]).to_csv(FEEDBACK_FILE, index=False)

def save_feedback(query, response, rating):
    feedback = {
        "timestamp": datetime.now().isoformat(),
        "query": query,
        "response": response,
        "rating": rating
    }
    pd.DataFrame([feedback]).to_csv(FEEDBACK_FILE, mode="a", header=False, index=False)

def response_generator(prompt, chain):
    response = chain.invoke(prompt)
    for word in response.split():
        yield word + " "
        time.sleep(0.05)

# File handling setup
save_directory = "docs"
save_path = "docs/file.pdf"
Path(save_directory).mkdir(parents=True, exist_ok=True)

st.title("📝 InsureAgent")

with st.sidebar:
    uploaded_file = st.file_uploader("Upload a document", type=("pdf"))
    if uploaded_file is not None:
        with open(save_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.success(f"File saved successfully: {save_path}")
    
    # Show feedback data toggle
    show_feedback = st.checkbox("Show feedback data")
    
    # Process documents
    retriever = process_docs(save_path)
    chain, chain_with_sources = create_rag_chain(retriever)

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages
for idx, message in enumerate(st.session_state.messages):
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        
        # Add rating buttons for assistant messages
        if message["role"] == "assistant":
            if "rating" not in message:
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("👍 Good", key=f"good_{idx}"):
                        message["rating"] = "good"
                        query = st.session_state.messages[idx-1]["content"]
                        save_feedback(query, message["content"], "good")
                        st.rerun()
                with col2:
                    if st.button("👎 Bad", key=f"bad_{idx}"):
                        message["rating"] = "bad"
                        query = st.session_state.messages[idx-1]["content"]
                        save_feedback(query, message["content"], "bad")
                        st.rerun()
            else:
                st.write(f"Rated: {message['rating'].capitalize()}")

# Show feedback data in sidebar if enabled
if show_feedback:
    st.sidebar.subheader("User Feedback")
    try:
        feedback_df = pd.read_csv(FEEDBACK_FILE)
        st.sidebar.dataframe(feedback_df)
        
        # Download button for feedback data
        csv = feedback_df.to_csv(index=False).encode('utf-8')
        st.sidebar.download_button(
            label="Download feedback as CSV",
            data=csv,
            file_name="feedback_data.csv",
            mime="text/csv"
        )
    except FileNotFoundError:
        st.sidebar.warning("No feedback data yet")

# Chat input and processing
if prompt := st.chat_input("Ask about your insurance document:"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate and display assistant response
    with st.chat_message("assistant"):
        response = st.write_stream(response_generator(prompt, chain))
    
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})