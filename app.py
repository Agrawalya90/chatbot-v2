import streamlit as st
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from transformers import pipeline
import faiss
import numpy as np
from supabase import create_client, Client
from dotenv import load_dotenv

# Load Supabase credentials
SUPABASE_URL = "https://wmroscjdhemybxznwmfg.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Indtcm9zY2pkaGVteWJ4em53bWZnIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NTU1MzY0OTgsImV4cCI6MjA3MTExMjQ5OH0.RaK3ROSAzD9-eeDVW28fkZGPS9i53bycJM9T88GlPRU"

if not SUPABASE_URL or not SUPABASE_KEY:
    st.error("Missing Supabase credentials. Add them to your .env file.")
else:
    supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

st.title("🧾 Local PDF Q&A with Supabase Storage (No API)")

uploaded_pdf = st.file_uploader("Upload a PDF", type="pdf")

if uploaded_pdf:
    file_name = uploaded_pdf.name
    file_data = uploaded_pdf.read()

    # Upload PDF to Supabase storage
    st.write("📤 Uploading to Supabase...")
    try:
        supabase.storage.from_("pdfs").upload(file_name, file_data)
        st.success(f"✅ Uploaded {file_name} to Supabase storage.")
    except Exception as e:
        st.error(f"Supabase upload failed: {e}")

    # Read text from PDF
    reader = PdfReader(uploaded_pdf)
    text = " ".join([page.extract_text() for page in reader.pages if page.extract_text()])

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = splitter.split_text(text)

    embedder = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = embedder.encode(chunks)

    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(np.array(embeddings))

    summarizer = pipeline("text2text-generation", model="google/flan-t5-small")

    st.session_state.update({
        "chunks": chunks,
        "index": index,
        "embedder": embedder,
        "summarizer": summarizer
    })

query = st.text_input("Ask a question about the PDF:")

if query and "index" in st.session_state:
    q_emb = st.session_state["embedder"].encode([query])
    D, I = st.session_state["index"].search(np.array(q_emb), k=3)
    results = [st.session_state["chunks"][i] for i in I[0]]
    context = " ".join(results)

    # Generate answer
    answer = st.session_state["summarizer"](
        f"Answer this question based on the context:\nQuestion: {query}\nContext: {context}",
        max_length=150,
        min_length=30,
        do_sample=False
    )[0]["generated_text"]

    st.subheader("💬 Answer:")
    st.success(answer)
