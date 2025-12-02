import streamlit as st
import PyPDF2
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from groq import Groq

client = Groq(api_key=st.secrets["GROQ_API_KEY"])

def extract_pdf_text(pdf_file):
    reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page in reader.pages:
        if page.extract_text():
            text += page.extract_text() + "\n"
    return text

def chunk_text(text, size=400):
    words = text.split()
    return [" ".join(words[i:i+size]) for i in range(0, len(words), size)]

embedder = SentenceTransformer("all-MiniLM-L6-v2")

def get_embeddings(texts):
    return np.array(embedder.encode(texts)).astype("float32")

def build_faiss(embeddings):
    dims = embeddings.shape[1]
    index = faiss.IndexFlatL2(dims)
    index.add(embeddings)
    return index

def search_faiss(query, chunks, chunk_emb, index, top_k=3):
    q_emb = embedder.encode([query]).astype("float32")
    dist, idxs = index.search(q_emb, top_k)
    return [chunks[i] for i in idxs[0]]

def groq_answer(question, context):
    prompt = f"""
Use ONLY this context:

{context}

Question: {question}
"""
    res = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role":"user","content":prompt}],
        max_tokens=250,
        temperature=0
    )
    return res.choices[0].message["content"]

st.title("📘 Simple University FAQ RAG Chatbot")

pdf = st.file_uploader("Upload your FAQ PDF", type="pdf")

if pdf:
    text = extract_pdf_text(pdf)
    chunks = chunk_text(text)
    chunk_emb = get_embeddings(chunks)
    index = build_faiss(chunk_emb)

    question = st.text_input("Ask a question:")

    if question:
        retrieved = search_faiss(question, chunks, chunk_emb, index)
        context = "\n\n".join(retrieved)
        answer = groq_answer(question, context)

        st.subheader("Answer")
        st.write(answer)

        st.subheader("Retrieved Chunks")
        for i, c in enumerate(retrieved):
            with st.expander(f"Chunk {i+1}"):
                st.write(c)
