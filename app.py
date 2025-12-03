# import streamlit as st
# import PyPDF2
# import numpy as np
# import faiss
# from groq import Groq

# # Load Groq API key from Streamlit Secrets
# client = Groq(api_key=st.secrets["GROQ_API_KEY"])

# # ----------------------------
# # PDF TEXT EXTRACTION
# # ----------------------------
# def extract_pdf_text(pdf_file):
#     reader = PyPDF2.PdfReader(pdf_file)
#     text = ""
#     for page in reader.pages:
#         if page.extract_text():
#             text += page.extract_text() + "\n"
#     return text

# # ----------------------------
# # TEXT CHUNKING
# # ----------------------------
# def chunk_text(text, size=300):
#     words = text.split()
#     return [" ".join(words[i:i+size]) for i in range(0, len(words), size)]

# # ----------------------------
# # GROQ EMBEDDINGS (CORRECT MODEL)
# # ----------------------------
# # def embed_text_list(text_list):
# #     response = client.embeddings.create(
# #         model="text-embedding-3-large",
# #         input=text_list
# #     )
# #     return np.array([item.embedding for item in response.data]).astype("float32")

# import re
# import json

# def embed_text_list(text_list):
#     vectors = []
#     for text in text_list:
#         resp = client.chat.completions.create(
#             model="llama-3.1-8b-instant",
#             messages=[
#                 {
#                     "role": "user",
#                     "content": f"""Convert the following text into a 256-dimensional numeric embedding.
#                     Output ONLY a JSON list of 256 numbers:

#                     {text}"""
#                 }
#             ],
#             max_tokens=300,
#             temperature=0
#         )

#         raw_output = resp.choices[0].message.content.strip()

#         # Extract JSON array using regex
#         match = re.search(r"\[.*\]", raw_output, re.DOTALL)
#         if match:
#             vector = json.loads(match.group(0))
#             vectors.append(vector)
#         else:
#             raise ValueError(f"Invalid response: {raw_output}")

#     return np.array(vectors).astype("float32")



# # ----------------------------
# # FAISS INDEX
# # ----------------------------
# def build_faiss(embeddings):
#     dims = embeddings.shape[1]
#     index = faiss.IndexFlatL2(dims)
#     index.add(embeddings)
#     return index

# def search_faiss(query, chunks, index, top_k=3):
#     q_emb = embed_text_list([query])
#     dist, idxs = index.search(q_emb, top_k)
#     return [chunks[i] for i in idxs[0]]

# # ----------------------------
# # GROQ LLM ANSWERING
# # ----------------------------
# def groq_answer(question, context):
#     prompt = f"""
# You are a helpful FAQ assistant.

# Use the retrieved context ONLY IF it is relevant.
# If context is irrelevant, answer the question generally.

# Context:
# {context}

# Question: {question}
# """
#     res = client.chat.completions.create(
#         model="llama-3.1-8b-instant",
#         messages=[{"role": "user", "content": prompt}],
#         max_tokens=250,
#         temperature=0.2
#     )

#     return res.choices[0].message.content

# # ----------------------------
# # STREAMLIT UI
# # ----------------------------
# st.title("📘 University FAQ RAG Chatbot (Groq + FAISS)")
# st.write("Upload your PDF and ask any question. Combines RAG + General LLM Answering.")

# pdf = st.file_uploader("Upload your FAQ PDF", type="pdf")

# if pdf:
#     st.success("PDF uploaded successfully!")

#     text = extract_pdf_text(pdf)
#     chunks = chunk_text(text)
#     chunk_embeddings = embed_text_list(chunks)
#     index = build_faiss(chunk_embeddings)

#     question = st.text_input("Ask a question:")

#     if question:
#         retrieved = search_faiss(question, chunks, index)
#         context = "\n\n".join(retrieved)

#         answer = groq_answer(question, context)

#         st.subheader("🟦 Answer")
#         st.write(answer)

#         st.subheader("🔍 Retrieved Chunks Used")
#         for i, c in enumerate(retrieved):
#             with st.expander(f"Chunk {i+1}"):
#                 st.write(c)

# ------------------

import streamlit as st
import PyPDF2
import numpy as np
import faiss
from gensim.models import KeyedVectors
from nltk.tokenize import word_tokenize
import nltk
from sklearn.decomposition import PCA

nltk.download('punkt')

# ----------------------------
# LOAD WORD2VEC MODEL
# ----------------------------
@st.cache_resource
def load_word2vec_model():
    # Download the model if needed: GoogleNews-vectors-negative300.bin
    return KeyedVectors.load_word2vec_format("GoogleNews-vectors-negative300.bin", binary=True)

model = load_word2vec_model()

# ----------------------------
# PDF TEXT EXTRACTION
# ----------------------------
def extract_pdf_text(pdf_file):
    reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
    return text.strip()

# ----------------------------
# TEXT CHUNKING
# ----------------------------
def chunk_text(text, size=300):
    words = text.split()
    return [" ".join(words[i:i+size]) for i in range(0, len(words), size)]

# ----------------------------
# WORD2VEC EMBEDDINGS
# ----------------------------
def embed_text_list(text_list, model, pca_model=None):
    vectors = []
    for text in text_list:
        tokens = [w for w in word_tokenize(text.lower()) if w in model.key_to_index]
        if tokens:
            vec = np.mean([model[w] for w in tokens], axis=0)
        else:
            vec = np.zeros(model.vector_size)
        vectors.append(vec)
    vectors = np.array(vectors, dtype=np.float32)
    
    # Reduce to 256 dimensions if PCA model is provided
    if pca_model:
        vectors = pca_model.transform(vectors)
    return vectors

# ----------------------------
# FAISS INDEX
# ----------------------------
def build_faiss(embeddings):
    dims = embeddings.shape[1]
    index = faiss.IndexFlatL2(dims)
    index.add(embeddings)
    return index

def search_faiss(query, chunks, index, model, pca_model=None, top_k=3):
    q_emb = embed_text_list([query], model, pca_model)
    dist, idxs = index.search(q_emb, top_k)
    return [chunks[i] for i in idxs[0]]

# ----------------------------
# STREAMLIT UI
# ----------------------------
st.title("📘 University FAQ RAG Chatbot (Word2Vec + FAISS)")
st.write("Upload your PDF and ask questions. Uses RAG with Word2Vec embeddings.")

pdf = st.file_uploader("Upload your FAQ PDF", type="pdf")

if pdf:
    st.success("PDF uploaded successfully!")
    text = extract_pdf_text(pdf)

    if len(text.strip()) == 0:
        st.warning("The PDF contains no extractable text.")
    else:
        chunks = chunk_text(text)
        
        # PCA for reducing Word2Vec 300-dim → 256-dim
        with st.spinner("Computing embeddings and PCA..."):
            temp_vectors = embed_text_list(chunks, model)
            pca = PCA(n_components=256)
            pca.fit(temp_vectors)
            chunk_embeddings = embed_text_list(chunks, model, pca)
        
        index = build_faiss(chunk_embeddings)

        question = st.text_input("Ask a question:")

        if question:
            with st.spinner("Searching relevant chunks..."):
                retrieved = search_faiss(question, chunks, index, model, pca)
            context = "\n\n".join(retrieved)

            st.subheader("🔍 Retrieved Chunks Used")
            for i, c in enumerate(retrieved):
                with st.expander(f"Chunk {i+1}"):
                    st.write(c)

