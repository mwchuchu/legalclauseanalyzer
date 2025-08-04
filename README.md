# maawaz-Legal-Assistant


# 🧠 Legal Assistant RAG Chatbot

The **Legal Assistant RAG Chatbot** is an AI-powered application designed to help users understand, query, and analyze legal documents or clauses. It uses **Retrieval-Augmented Generation (RAG)** to fetch relevant content from a document database and generate accurate, context-aware answers with the help of LLMs.

## 🚀 Features

- 📄 Upload and process legal documents (PDF, DOCX, TXT)
- 🧩 Automatic document chunking and vector embedding
- 🔍 Semantic search powered by FAISS
- 🤖 Context-aware responses via Gemini Pro / OpenAI / any pluggable LLM
- 📝 Tracks user queries and chatbot responses
- 🧠 Intent classification (e.g., "Summarize", "Explain", "Compare", etc.)
- 🗂️ Query history & chat log storage (SQL/MongoDB support)
- 🌐 Built with FastAPI and Celery (asynchronous background tasks)

---

## 📁 Project Structure


---

## 🛠️ Tech Stack

| Component       | Tool / Library                  |
|----------------|----------------------------------|
| Backend         | FastAPI                         |
| Asynchronous Tasks | Celery + Redis               |
| Vector Store    | FAISS                           |
| Embedding Model | SentenceTransformers            |
| LLM             | Gemini Pro / OpenAI GPT         |
| Storage         | PostgreSQL            |
| Authentication  |  JWT         |

---

## 🧰 Installation

1. **Clone the repository:**

```bash
git clone https://gitlab.com/divedeepai/maawaz-legal-assistant
cd legal-rag-chatbot
