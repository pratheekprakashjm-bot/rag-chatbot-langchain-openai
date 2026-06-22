import streamlit as st
import os
from rag_pipeline import RAGPipeline

st.set_page_config(page_title="RAG Chatbot", layout="wide")

st.title("📄 RAG Chatbot using LangChain + OpenAI")

rag = RAGPipeline()

uploaded_file = st.file_uploader("Upload PDF", type="pdf")

if uploaded_file:
    file_path = f"data/{uploaded_file.name}"

    with open(file_path, "wb") as f:
        f.write(uploaded_file.read())

    with st.spinner("Processing document..."):
        docs = rag.load_pdf(file_path)
        rag.process_documents(docs)
        rag.create_qa_chain()

    st.success("Document ready! Ask questions below 👇")

    question = st.text_input("Enter your question")

    if question:
        with st.spinner("Generating answer..."):
            response = rag.ask(question)

        st.subheader("Answer")
        st.write(response["result"])

        with st.expander("Source Documents"):
            for doc in response["source_documents"]:
                st.write(doc.page_content)
