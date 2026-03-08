# ===============================
# Imports
# ===============================

import os
import tempfile
import streamlit as st
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma


# ===============================
# Setup
# ===============================

load_dotenv()
st.write("ENV KEY:", os.getenv("GROQ_API_KEY"))

st.set_page_config(page_title="📝 RAG Q&A", layout="wide")
st.title("📝 RAG Q&A with Multiple PDFs + Chat History")


# ===============================
# Sidebar Config
# ===============================

with st.sidebar:
    st.header("⚙️ Config")
    api_key_input = st.text_input("Groq API Key", type="password")
    st.caption("Upload PDFs → Ask questions → Get Answers")


api_key = api_key_input or os.getenv("GROQ_API_KEY")

if not api_key:
    st.warning("Please enter your Groq API Key (or set GROQ_API_KEY in .env)")
    st.stop()


# ===============================
# Load Embeddings (cached)
# ===============================

@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        encode_kwargs={"normalize_embeddings": True}
    )

embeddings = load_embeddings()


# ===============================
# Initialize LLM
# ===============================

llm = ChatGroq(
    groq_api_key=api_key,
    model="llama-3.3-70b-versatile"
)


# ===============================
# Upload PDFs
# ===============================

uploaded_files = st.file_uploader(
    "📚 Upload PDF files",
    type="pdf",
    accept_multiple_files=True
)

if not uploaded_files:
    st.info("Please upload one or more PDFs to begin.")
    st.stop()


# ===============================
# Load PDFs (cached)
# ===============================

@st.cache_data
def load_pdfs(uploaded_files):

    all_docs = []
    tmp_paths = []

    for pdf in uploaded_files:

        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
        tmp.write(pdf.getvalue())
        tmp.close()

        tmp_paths.append(tmp.name)

        loader = PyPDFLoader(tmp.name)
        docs = loader.load()

        for d in docs:
            d.metadata["source_file"] = pdf.name

        all_docs.extend(docs)

    for p in tmp_paths:
        try:
            os.unlink(p)
        except Exception:
            pass

    return all_docs


all_docs = load_pdfs(uploaded_files)

st.success(f"✅ Loaded {len(all_docs)} pages from {len(uploaded_files)} PDFs")


# ===============================
# Text Chunking
# ===============================

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1200,
    chunk_overlap=120
)

splits = text_splitter.split_documents(all_docs)


# ===============================
# Vectorstore (cached)
# ===============================

INDEX_DIR = "chroma_index"

@st.cache_resource
def create_vectorstore(splits):

    return Chroma.from_documents(
        splits,
        embeddings,
        persist_directory=INDEX_DIR
    )

vectorstore = create_vectorstore(splits)


retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 5, "fetch_k": 20}
)

st.sidebar.write(f"🔍 Indexed {len(splits)} chunks for retrieval")


# ===============================
# Helper Function
# ===============================

def _join_docs(docs, max_chars=7000):

    chunks = []
    total = 0

    for d in docs:

        piece = d.page_content

        if total + len(piece) > max_chars:
            break

        chunks.append(piece)
        total += len(piece)

    return "\n\n---\n\n".join(chunks)


# ===============================
# Prompts
# ===============================

contextualize_q_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "Rewrite the user's latest question into a standalone search query "
        "using the chat history for context. Return only the rewritten query."
    ),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}")
])


qa_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a STRICT RAG assistant. You must answer using ONLY the provided context.\n"
        "If the context does NOT contain the answer, reply exactly:\n"
        "'Out of scope - not found in provided documents.'\n\n"
        "Context:\n{context}"
    ),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}")
])


# ===============================
# Chat History Storage
# ===============================

if "chathistory" not in st.session_state:
    st.session_state.chathistory = {}


def get_history(session_id):

    if session_id not in st.session_state.chathistory:
        st.session_state.chathistory[session_id] = ChatMessageHistory()

    return st.session_state.chathistory[session_id]


# ===============================
# Chat UI
# ===============================

session_id = st.text_input("🆔 Session ID", value="default_session")

user_q = st.chat_input("💬 Ask a question...")


# ===============================
# Chat Logic
# ===============================

if user_q:

    history = get_history(session_id)

    # Rewrite query using history
    rewrite_msgs = contextualize_q_prompt.format_messages(
        chat_history=history.messages,
        input=user_q
    )

    standalone_q = llm.invoke(rewrite_msgs).content.strip()

    # Retrieve documents
    docs = retriever.invoke(standalone_q)

    if not docs:

        answer = "Out of scope - not found in provided documents."

        st.chat_message("user").write(user_q)
        st.chat_message("assistant").write(answer)

        history.add_user_message(user_q)
        history.add_ai_message(answer)

        st.stop()

    # Build context
    context_str = _join_docs(docs)

    # Ask LLM
    qa_msgs = qa_prompt.format_messages(
        chat_history=history.messages,
        input=user_q,
        context=context_str
    )

    answer = llm.invoke(qa_msgs).content


    # Display messages
    st.chat_message("user").write(user_q)
    st.chat_message("assistant").write(answer)

    history.add_user_message(user_q)
    history.add_ai_message(answer)


    # ===============================
    # Debug Panels
    # ===============================

    with st.expander("🧪 Debug: Query + Retrieval"):

        st.write("**Standalone query:**")
        st.code(standalone_q)

        st.write(f"Retrieved {len(docs)} chunks")


    with st.expander("📑 Retrieved Chunks"):

        for i, doc in enumerate(docs, 1):

            st.markdown(
                f"**{i}. {doc.metadata.get('source_file','Unknown')} "
                f"(page {doc.metadata.get('page','?')})**"
            )

            text = doc.page_content[:500]

            if len(doc.page_content) > 500:
                text += "..."

            st.write(text)
