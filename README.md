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
