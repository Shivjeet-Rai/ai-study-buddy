import streamlit as st
from pypdf import PdfReader
import os
import google.generativeai as genai

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


# GEMINI KEY
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))


# PAGE CONFIG
st.set_page_config(
    page_title="AI Study Buddy",
    page_icon="📘",
    layout="centered"
)

st.title("📘 AI Study Buddy")
st.caption("Upload your study notes and ask questions powered by AI")

st.divider()


# -------------------- FILE UPLOAD --------------------
st.subheader("📄 Upload Notes")

uploaded_file = st.file_uploader(
    "Upload a PDF file",
    type="pdf",
    help="Only text-based PDFs are supported"
)


if uploaded_file is not None:
    with st.spinner("Reading PDF..."):
        reader = PdfReader(uploaded_file)

        text = ""
        for page in reader.pages:
            extracted = page.extract_text()
            if extracted:
                text += extracted

    st.success("PDF uploaded and processed successfully!")

    # -------------------- CHUNKING --------------------
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )
    chunks = splitter.split_text(text)

    # -------------------- EMBEDDINGS + VECTOR STORE --------------------
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vector_store = FAISS.from_texts(chunks, embeddings)

    st.divider()

    # -------------------- QUESTION INPUT --------------------
    st.subheader("❓ Ask a Question")

    query = st.text_input(
        "Type your question here",
        placeholder="e.g. What is an event horizon?"
    )

    col1, col2 = st.columns([1, 1])

    with col1:
        ask_button = st.button("🔍 Get Answer")

    with col2:
        reset_button = st.button("🔄 Reset")

    if reset_button:
        st.rerun()

    if ask_button and query:
        with st.spinner("Thinking..."):
            results = vector_store.similarity_search(query, k=3)
            context = "\n\n".join([doc.page_content for doc in results])

            model = genai.GenerativeModel("models/gemini-2.5-flash-lite")

            prompt = f"""
You are a study assistant.
Answer the question using ONLY the context below.
If the answer is not in the context, say "I don't know".

Context:
{context}

Question:
{query}
"""

            response = model.generate_content(prompt)

        st.divider()

        # -------------------- ANSWER --------------------
        st.subheader("✅ Answer")
        st.write(response.text)

    elif ask_button and not query:
        st.warning("Please enter a question before clicking Get Answer.")
