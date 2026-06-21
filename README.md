# 📄 PDF Q&A Chatbot

An AI-powered chatbot that lets you upload any PDF and ask questions about it in natural language. Answers come **only from your document** — no hallucination.

> 🚀 **Live Demo:** [your-app-url.streamlit.app](https://your-app-url.streamlit.app)

---

## ✨ Features

- 📂 Upload any PDF and chat with it instantly
- 🧠 RAG architecture — answers grounded in your document only
- 💬 Multi-turn conversation with memory
- 📎 Source page citations for every answer
- ⚡ Local embeddings — no API timeouts
- 🎨 Clean dark UI

---

## 🛠️ Tech Stack

| Component | Tool |
|---|---|
| UI | Streamlit |
| LLM | Google Gemini 2.5 Flash Lite |
| Embeddings | HuggingFace `all-MiniLM-L6-v2` (local) |
| Vector DB | FAISS |
| Orchestration | LangChain |
| PDF Reading | PyPDF |

---

## 🧠 Architecture

```
PDF Upload → Text Extraction → Chunking → Local Embeddings → FAISS Index
                                                                    ↓
User Question → Embed Question → FAISS finds top 4 chunks → Gemini answers
```

---

## 🚀 Local Setup

```bash
# Clone the repo
git clone https://github.com/rakeshricky442/pdf-chatbot.git
cd pdf-chatbot

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Mac/Linux

# Install dependencies
pip install -r requirements.txt
pip install sentence-transformers

# Add your Gemini API key
mkdir .streamlit
echo 'GEMINI_API_KEY = "your_key_here"' > .streamlit/secrets.toml

# Run the app
streamlit run app.py
```

Get a free Gemini API key at **[aistudio.google.com/app/apikey](https://aistudio.google.com/app/apikey)**

---

## ☁️ Deploy on Streamlit Cloud

1. Fork this repo
2. Go to **[share.streamlit.io](https://share.streamlit.io)**
3. Connect your GitHub repo
4. Add `GEMINI_API_KEY` in Secrets settings
5. Deploy!

---

## 📁 Project Structure

```
pdf-chatbot/
├── app.py              # Main application
├── requirements.txt    # Dependencies
├── .env.example        # API key template
└── README.md
```

---

## 👤 Author

**Rakesh** 
🌐 [Portfolio](https://rakeshricky442.github.io) · [LinkedIn](#) · [GitHub](https://github.com/rakeshricky442)
