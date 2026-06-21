import streamlit as st
import os
import time
import tempfile
from dotenv import load_dotenv

# ─── LangChain imports ────────────────────────────────────────────────────────
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

# ─── Load environment variables ───────────────────────────────────────────────
load_dotenv()

# ══════════════════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="PDF Chat",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════════════════════════════════════
# CUSTOM CSS
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap');
 
:root {
    --bg:        #0f1117;
    --surface:   #1a1d27;
    --border:    #2a2d3a;
    --accent:    #4f8ef7;
    --accent-dim:#1e3a6e;
    --text:      #e2e8f0;
    --muted:     #8892a4;
    --user-bg:   #1e3a6e;
    --bot-bg:    #1a1d27;
    --success:   #22c55e;
    --font:      'Inter', sans-serif;
}
 
html, body, [class*="css"] { font-family: var(--font); }
.stApp { background: var(--bg); color: var(--text); }
 
[data-testid="stSidebar"] {
    background: var(--surface) !important;
    border-right: 1px solid var(--border);
}
[data-testid="stSidebar"] * { color: var(--text) !important; }
 
.page-header {
    padding: 1.5rem 0 1rem;
    border-bottom: 1px solid var(--border);
    margin-bottom: 1.5rem;
}
.page-header h1 { font-size: 1.75rem; font-weight: 600; color: var(--text); margin: 0; }
.page-header p { color: var(--muted); margin: 0.3rem 0 0; font-size: 0.9rem; }
 
.status-badge {
    display: inline-flex; align-items: center; gap: 0.4rem;
    background: #14291a; border: 1px solid #166534; color: var(--success);
    padding: 0.3rem 0.8rem; border-radius: 999px; font-size: 0.8rem;
    font-weight: 500; margin-bottom: 1rem;
}
 
.chat-wrapper { display: flex; flex-direction: column; gap: 1rem; margin-bottom: 1.5rem; }
.msg { display: flex; gap: 0.75rem; align-items: flex-start; max-width: 85%; }
.msg.user { margin-left: auto; flex-direction: row-reverse; }
.msg.bot { margin-right: auto; }
.avatar {
    width: 32px; height: 32px; border-radius: 8px;
    display: flex; align-items: center; justify-content: center;
    font-size: 1rem; flex-shrink: 0;
}
.avatar.user { background: var(--accent-dim); }
.avatar.bot { background: #2a1f4a; }
.bubble {
    padding: 0.75rem 1rem; border-radius: 12px;
    font-size: 0.92rem; line-height: 1.6; border: 1px solid var(--border);
}
.bubble.user { background: var(--user-bg); border-color: #2a4a8a; color: #c8d9f5; border-radius: 12px 4px 12px 12px; }
.bubble.bot { background: var(--bot-bg); border-color: var(--border); color: var(--text); border-radius: 4px 12px 12px 12px; }
 
.stTextInput input {
    background: var(--surface) !important; border: 1px solid var(--border) !important;
    border-radius: 10px !important; color: var(--text) !important;
    font-size: 0.95rem !important; padding: 0.75rem 1rem !important;
}
.stTextInput input:focus {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 3px rgba(79,142,247,0.15) !important;
}
 
.stButton > button {
    background: var(--accent) !important; color: #fff !important;
    border: none !important; border-radius: 8px !important;
    font-weight: 500 !important; padding: 0.5rem 1.25rem !important;
}
.stButton > button:hover { opacity: 0.85 !important; }
 
.empty-state { text-align: center; padding: 3rem 1rem; color: var(--muted); }
.empty-state .icon { font-size: 3rem; margin-bottom: 1rem; }
.empty-state h3 { color: var(--text); font-size: 1.1rem; margin-bottom: 0.5rem; }
 
.info-box {
    background: #1a2235; border: 1px solid #2a4a8a; border-radius: 10px;
    padding: 0.85rem 1rem; font-size: 0.85rem; color: #93b4e8;
    margin-bottom: 1rem; line-height: 1.5;
}
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def get_api_key() -> str:
    try:
        if "GEMINI_API_KEY" in st.secrets:
            return st.secrets["GEMINI_API_KEY"]
    except Exception:
        pass
    env_key = os.getenv("GEMINI_API_KEY", "")
    if env_key:
        return env_key
    return st.session_state.get("api_key_input", "")


@st.cache_resource(show_spinner=False)
def load_embeddings():
    """
    Load HuggingFace embedding model — runs locally on your Mac.
    No API key needed. Downloads once (~90MB), then cached forever.
    """
    return HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"}
    )


@st.cache_resource(show_spinner=False)
def build_vectorstore(pdf_bytes: bytes, pdf_name: str):
    """
    Converts PDF into FAISS vector database using local HuggingFace embeddings.
    No API calls = no timeouts!
    """

    # Step 1: Save PDF to temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(pdf_bytes)
        tmp_path = tmp.name

    # Step 2: Load PDF pages
    loader = PyPDFLoader(tmp_path)
    pages = loader.load()

    # Step 3: Split into chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = splitter.split_documents(pages)

    # Step 4: Create FAISS vectorstore using local embeddings
    embeddings = load_embeddings()
    vectorstore = FAISS.from_documents(chunks, embeddings)

    # Clean up
    os.unlink(tmp_path)

    return vectorstore


def build_qa_chain(vectorstore, api_key: str):
    """
    Creates QA chain — Gemini answers questions based on PDF chunks from FAISS.
    """

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash-lite",
        google_api_key=api_key,
        temperature=0.2,
        convert_system_message_to_human=True
    )

    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="answer"
    )

    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 4}
    )

    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        return_source_documents=True,
        verbose=False
    )
    return chain


def render_chat_history():
    if not st.session_state.messages:
        st.markdown("""
        <div class="empty-state">
            <div class="icon">💬</div>
            <h3>Ready to answer your questions</h3>
            <p>Ask anything about your PDF!</p>
        </div>
        """, unsafe_allow_html=True)
        return

    st.markdown('<div class="chat-wrapper">', unsafe_allow_html=True)
    for msg in st.session_state.messages:
        role = msg["role"]
        content = msg["content"]
        sources = msg.get("sources", [])

        if role == "user":
            st.markdown(f"""
            <div class="msg user">
                <div class="avatar user">🙋</div>
                <div class="bubble user">{content}</div>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="msg bot">
                <div class="avatar bot">🤖</div>
                <div class="bubble bot">{content}</div>
            </div>""", unsafe_allow_html=True)

            if sources:
                with st.expander("📎 View sources from PDF", expanded=False):
                    for i, src in enumerate(sources, 1):
                        page_num = src.metadata.get("page", "?") + 1
                        st.markdown(f"**Source {i} — Page {page_num}**")
                        st.caption(src.page_content[:400] + "...")
    st.markdown('</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# SESSION STATE
# ══════════════════════════════════════════════════════════════════════════════
if "messages" not in st.session_state:
    st.session_state.messages = []
if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None
if "pdf_processed" not in st.session_state:
    st.session_state.pdf_processed = False
if "current_pdf" not in st.session_state:
    st.session_state.current_pdf = None


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## ⚙️ Setup")
    st.markdown("---")

    api_key = get_api_key()
    if not api_key:
        st.markdown("**🔑 Gemini API Key**")
        typed_key = st.text_input(
            "Paste your API key",
            type="password",
            placeholder="AIza...",
            key="api_key_input",
        )
        if typed_key:
            api_key = typed_key
        st.markdown(
            "<div class='info-box'>🆓 Get a free key at "
            "<a href='https://aistudio.google.com/app/apikey' target='_blank' "
            "style='color:#93b4e8'>aistudio.google.com</a></div>",
            unsafe_allow_html=True
        )
    else:
        st.markdown("<div class='status-badge'>✅ API Key loaded</div>", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("**📄 Upload PDF**")
    uploaded_file = st.file_uploader(
        "Choose a PDF file",
        type=["pdf"],
        label_visibility="collapsed"
    )

    if uploaded_file and api_key:
        if uploaded_file.name != st.session_state.current_pdf:
            if st.button("🔍 Process PDF", use_container_width=True):
                with st.spinner("Reading and indexing PDF…"):
                    try:
                        pdf_bytes = uploaded_file.read()
                        vectorstore = build_vectorstore(pdf_bytes, uploaded_file.name)
                        st.session_state.qa_chain = build_qa_chain(vectorstore, api_key)
                        st.session_state.pdf_processed = True
                        st.session_state.current_pdf = uploaded_file.name
                        st.session_state.messages = []
                        st.success(f"✅ Ready! Ask questions about **{uploaded_file.name}**")
                    except Exception as e:
                        st.error(f"Error processing PDF: {str(e)}")
        else:
            st.markdown(
                f"<div class='status-badge'>✅ {uploaded_file.name} loaded</div>",
                unsafe_allow_html=True
            )
    elif uploaded_file and not api_key:
        st.warning("⚠️ Enter your API key above first.")

    st.markdown("---")

    if st.session_state.pdf_processed:
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

    with st.expander("💡 Tips for better answers"):
        st.markdown("""
        - Try: *"Summarize this document"*
        - Try: *"What are my technical skills?"*
        - Try: *"List all projects mentioned"*
        - Works best with text-based PDFs
        """)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN AREA
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="page-header">
    <h1>📄 PDF Chat</h1>
    <p>Upload any PDF and ask questions — answers come only from your document.</p>
</div>
""", unsafe_allow_html=True)

if not st.session_state.pdf_processed:
    st.markdown("""
    <div class="empty-state">
        <div class="icon">📂</div>
        <h3>No PDF loaded yet</h3>
        <p>Upload a PDF in the sidebar and click <strong>Process PDF</strong> to begin.</p>
    </div>
    """, unsafe_allow_html=True)
else:
    render_chat_history()

    col1, col2 = st.columns([5, 1])
    with col1:
        question = st.text_input(
            "Ask a question",
            placeholder="e.g. What are my technical skills?",
            label_visibility="collapsed",
            key="question_input"
        )
    with col2:
        send = st.button("Send →", use_container_width=True)

    if (send or question) and question.strip():
        if not st.session_state.qa_chain:
            st.error("Please process a PDF first.")
        else:
            st.session_state.messages.append({"role": "user", "content": question})

            with st.spinner("Searching PDF and generating answer…"):
                try:
                    result = st.session_state.qa_chain({
                        "question": question,
                        "chat_history": []
                    })
                    answer = result.get("answer", "I couldn't find an answer in the PDF.")
                    sources = result.get("source_documents", [])
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "sources": sources
                    })
                except Exception as e:
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": f"⚠️ Error: {str(e)}",
                        "sources": []
                    })

            st.rerun()