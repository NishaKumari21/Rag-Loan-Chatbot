import streamlit as st
from retriever import Retriever
from generator import Generator

st.set_page_config(page_title="Loan RAG Chatbot")
st.title("📄 Loan Approval Chatbot (RAG)")

query = st.text_input("Ask a question about loan approvals")

if 'retriever' not in st.session_state:
    retriever = Retriever()
    retriever.prepare_documents()
    retriever.create_index()
    st.session_state.retriever = retriever

if 'generator' not in st.session_state:
    st.session_state.generator = Generator(use_openai=True)  # Change to False if using HF

if query:
    context = st.session_state.retriever.retrieve_context(query)
    response = st.session_state.generator.generate_response(context, query)
    st.markdown("**Answer:**")
    st.write(response)