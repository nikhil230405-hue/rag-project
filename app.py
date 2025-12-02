import streamlit as st
import PyPDF2
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from groq import Groq

# Load Groq API key from Streamlit Secrets
client = Groq(api_key=st.secrets["GROQ_API_KEY"])

# ----------------------------
# PDF TEXT EXTRACTION
# ----------------------------
def extract_pdf_text(pdf_file):
    reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page in reader.pages:
        if page.extract_text():
            text += page.extract_text() + "\n"
    return text

# ----------------------------
# TEXT CHUNKING
# ----------------------------
def chunk_text(text, size=400):
    words = text.split()
    return [" ".join(words[i:i+size]) for i in range(0, len(words), size)]

# ----------------------------
# EMBEDDING MODEL
# ----------------------------
embedder = SentenceTransformer("all-MiniLM-L6-v2")

def get_embeddings(texts):
    return np.array(embedder.encode(texts)).astype("float32")

# ----------------------------
# FAISS INDEX
# ----------------------------
def build_faiss(embeddings):
    dims = embeddings.shape[1]
    index = faiss.IndexFlatL2(dims)
    index.add(embeddings)
    return index

def search_faiss(query, chunks, chunk_emb, index, top_k=3):
    q_emb = embedder.encode([query]).astype("float32")
    dist, idxs = index.search(q_emb, top_k)
    return [chunks[i] for i in idxs[0]]

# ----------------------------
# GROQ LLM ANSWER
# ----------------------------
def groq_answer(question, context):
    prompt = f"""
Use ONLY this context to answer the question:

{context}

Question: {question}
"""
    res = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=250,
        temperature=0
    )

    # FIX: Groq returns message.content not message["content"]
    return res.choices[0].message.content

# ----------------------------
# STREAMLIT UI
# ----------------------------
st.title("📘 Simple University FAQ RAG Chatbot (Groq + FAISS)")
st.write("Upload your PDF and ask any question.")

pdf = st.file_uploader("Upload your FAQ PDF", type="pdf")

if pdf:
    st.success("PDF uploaded successfully!")

    text = extract_pdf_text(pdf)
    chunks = chunk_text(text)
    chunk_emb = get_embeddings(chunks)
    index = build_faiss(chunk_emb)

    question = st.text_input("Ask a question:")

    if question:
        retrieved = search_faiss(question, chunks, chunk_emb, index)
        context = "\n\n".join(retrieved)

        answer = groq_answer(question, context)

        st.subheader("🟦 Answer")
        st.write(answer)

        st.subheader("🔍 Retrieved Chunks Used")
        for i, c in enumerate(retrieved):
            with st.expander(f"Chunk {i+1}"):
                st.write(c)
