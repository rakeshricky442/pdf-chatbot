# 📄 PDF Q&A Chatbot

An AI-powered chatbot that lets you upload any PDF and ask questions about it in natural language. Built with Google Gemini API, LangChain, FAISS, and Streamlit.

---

## 🧠 How It Works (Architecture)

```
User uploads PDF
      ↓
PyPDFLoader reads PDF pages
      ↓
Text splitter breaks pages into 1000-char chunks (with 200-char overlap)
      ↓
Gemini Embeddings converts each chunk → numerical vector (embedding)
      ↓
FAISS stores all embeddings in a searchable index
      ↓
User asks a question
      ↓
Question → converted to embedding → FAISS finds top 4 similar chunks
      ↓
[Question + matching chunks + chat history] → sent to Gemini Flash
      ↓
Gemini reads ONLY those chunks and answers
      ↓
Answer shown to user with source citations
```

**Key concept:** The chatbot NEVER answers from its general knowledge — it only reads your PDF chunks. This is called **Retrieval-Augmented Generation (RAG)**.

---

## 🗂️ Project Structure

```
pdf-chatbot/
├── app.py                  # Main Streamlit application (all logic here)
├── requirements.txt        # Python dependencies
├── .env.example            # Template for your API key (local dev)
├── .gitignore              # Files to never commit to GitHub
├── .streamlit/
│   └── secrets.toml        # Template for Streamlit Cloud secrets
└── README.md               # This file
```

---

## 🔑 Step 1: Get a Free Gemini API Key

1. Go to **[aistudio.google.com/app/apikey](https://aistudio.google.com/app/apikey)**
2. Sign in with your Google account
3. Click **"Create API Key"**
4. Copy the key (starts with `AIza...`)

**Free tier limits (as of 2024):**
- 15 requests per minute
- 1 million tokens per minute  
- 1,500 requests per day
- No credit card required

---

## 💻 Step 2: Local Setup (Run on Your Computer)

### 2.1 — Create a virtual environment

```bash
# Navigate to project folder
cd pdf-chatbot

# Create virtual environment
python -m venv venv

# Activate it
# On Windows:
venv\Scripts\activate
# On Mac/Linux:
source venv/bin/activate
```

### 2.2 — Install dependencies

```bash
pip install -r requirements.txt
```

This installs: Streamlit, LangChain, Google Gemini SDK, FAISS, PyPDF, and dotenv.

### 2.3 — Set up your API key

```bash
# Copy the example file
cp .env.example .env

# Open .env and replace the placeholder with your real key
# GEMINI_API_KEY=AIzaSy...your_real_key_here
```

### 2.4 — Run the app

```bash
streamlit run app.py
```

The app opens at **http://localhost:8501** in your browser.

### 2.5 — Use the app

1. Your API key loads automatically from `.env`
2. Upload any PDF using the sidebar
3. Click **"Process PDF"** — wait ~10–30 seconds
4. Ask questions in the chat box!

---

## ☁️ Step 3: Deploy Free on Streamlit Cloud

### 3.1 — Push your code to GitHub

```bash
# Initialize git (first time only)
git init
git add .
git commit -m "Initial commit: PDF Q&A Chatbot"

# Create a new repo on github.com, then:
git remote add origin https://github.com/YOUR_USERNAME/pdf-chatbot.git
git push -u origin main
```

> ⚠️ Make sure `.env` is in `.gitignore` — never push your API key to GitHub!

### 3.2 — Deploy on Streamlit Cloud

1. Go to **[share.streamlit.io](https://share.streamlit.io)**
2. Sign in with GitHub
3. Click **"New app"**
4. Select your repository → Branch: `main` → Main file: `app.py`
5. Click **"Advanced settings"** → **"Secrets"**
6. Paste this in the secrets box:
   ```toml
   GEMINI_API_KEY = "AIzaSy...your_actual_key"
   ```
7. Click **"Deploy"**

Your app gets a free public URL like: `https://your-username-pdf-chatbot-app-xxxx.streamlit.app`

---

## 🧪 Test Questions to Try

After uploading a research paper or report, try:
- *"Summarize this document in 3 bullet points"*
- *"What is the main methodology used?"*
- *"What are the key findings or conclusions?"*
- *"What does the author say about [topic]?"*
- *"List all the references mentioned"*

---

## ⚠️ Known Limitations

| Limitation | Reason | Workaround |
|------------|--------|------------|
| Scanned PDFs may not work | Images aren't read as text | Use text-based PDFs |
| Very large PDFs are slow | More chunks = more embedding calls | Try PDFs under 100 pages |
| Answers limited to PDF content | By design (RAG approach) | This is a feature, not a bug! |

---

## 📌 Resume Bullet Points

Use these in your **Projects** section:

### Option A — Concise (1 line)
> **PDF Q&A Chatbot** | Python, LangChain, FAISS, Google Gemini API, Streamlit  
> Built an end-to-end RAG pipeline that enables natural language Q&A over uploaded PDFs, using FAISS vector search and Gemini Flash for context-aware responses; deployed on Streamlit Cloud.

### Option B — Detailed (3 bullets, for a full project entry)
> **PDF Q&A Chatbot** | Python · LangChain · FAISS · Google Gemini API · Streamlit  
> - Engineered a Retrieval-Augmented Generation (RAG) pipeline using LangChain and FAISS to index PDF content as semantic embeddings, enabling accurate document-grounded Q&A without LLM hallucination  
> - Integrated Google Gemini 1.5 Flash API with conversational memory, allowing multi-turn follow-up questions while maintaining context across the chat session  
> - Deployed as an interactive web application on Streamlit Cloud with PDF upload, real-time answer streaming, and source citation display

### Option C — Skills-focused
> - Developed a RAG-based PDF chatbot using LangChain's ConversationalRetrievalChain, chunking documents with RecursiveCharacterTextSplitter and retrieving top-k relevant passages via FAISS similarity search before prompting Gemini Flash

---

## 🛠️ Tech Stack Summary (for interviews)

| Component | Tool | Why |
|-----------|------|-----|
| UI | Streamlit | Fast Python web apps, no HTML/CSS needed |
| LLM | Google Gemini 1.5 Flash | Free tier, fast, strong reasoning |
| Embeddings | Gemini `embedding-001` | Converts text to semantic vectors |
| Vector DB | FAISS | Fast similarity search (by Facebook AI) |
| Orchestration | LangChain | Connects all components in a pipeline |
| PDF reading | PyPDF | Extracts text from PDF pages |

---

## 💬 Interview Talking Points

**Q: What is RAG?**  
A: Retrieval-Augmented Generation. Instead of relying on the LLM's training data, we retrieve relevant chunks from a document and include them in the prompt. This makes answers accurate and document-specific.

**Q: Why FAISS?**  
A: FAISS (Facebook AI Similarity Search) stores text as vectors and finds the most semantically similar chunks to a query in milliseconds — even across thousands of pages.

**Q: Why chunk the PDF?**  
A: LLMs have token limits. Chunking lets us selectively feed only the most relevant parts of a large document, reducing cost and improving answer quality.

**Q: How did you handle conversational memory?**  
A: LangChain's `ConversationBufferMemory` stores prior Q&A pairs and includes them in each new prompt, so the model can handle follow-up questions like "tell me more about that."
