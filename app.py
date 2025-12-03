import streamlit as st
import PyPDF2
import numpy as np
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
def chunk_text(text, size=300):
    words = text.split()
    return [" ".join(words[i:i+size]) for i in range(0, len(words), size)]

# ----------------------------
# GROQ EMBEDDINGS
# ----------------------------
def embed_text_list(text_list):
    response = client.embeddings.create(
        model="nomic-embed-text",
        input=text_list
    )
    return np.array([item.embedding for item in response.data]).astype("float32")

# ----------------------------
# FAISS INDEX
# ----------------------------
def build_faiss(embeddings):
    dims = embeddings.shape[1]
    index = faiss.IndexFlatL2(dims)
    index.add(embeddings)
    return index

def search_faiss(query, chunks, index, top_k=3):
    q_emb = embed_text_list([query])
    dist, idxs = index.search(q_emb, top_k)
    return [chunks[i] for i in idxs[0]]

# ----------------------------
# GROQ LLM FOR ANSWERING
# ----------------------------
def groq_answer(question, context):
    prompt = f"""
You are a helpful FAQ assistant.

Use the provided context **ONLY IF RELEVANT**.  
If the context is not related to the question, give a general helpful answer.

Context:
{context}

Question: {question}
"""
    res = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=250,
        temperature=0.2
    )

    return res.choices[0].message.content

# ----------------------------
# STREAMLIT UI
# ----------------------------
st.title("📘 University FAQ RAG Chatbot (Groq + FAISS)")
st.write("Upload your PDF and ask any question — RAG + general LLM answers.")

pdf = st.file_uploader("Upload your FAQ PDF", type="pdf")

if pdf:
    st.success("PDF uploaded successfully!")

    # Build RAG system
    text = extract_pdf_text(pdf)
    chunks = chunk_text(text)
    chunk_embeddings = embed_text_list(chunks)
    index = build_faiss(chunk_embeddings)

    question = st.text_input("Ask a question:")

    if question:
        retrieved_chunks = search_faiss(question, chunks, index)
        context = "\n\n".join(retrieved_chunks)

        answer = groq_answer(question, context)

        st.subheader("🟦 Answer")
        st.write(answer)

        st.subheader("🟧 Retrieved Chunks Used")
        for i, c in enumerate(retrieved_chunks):
            with st.expander(f"Chunk {i+1}"):
                st.write(c)
