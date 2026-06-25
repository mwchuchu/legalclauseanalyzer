# maawaz-Legal-Assistant

# 🧠 Legal Assistant RAG Chatbot

The **Legal Assistant RAG Chatbot** is a FastAPI-based application for conversational search over legal clause data. It supports CSV/ZIP uploads, vector embedding storage using FAISS, async processing with Celery, and a chatbot interface powered by Google Gemini.

## 🚀 Features

- 📄 Upload CSV or ZIP files containing legal clauses
- ⚡ Asynchronous CSV/ZIP processing with Celery + Redis
- 🧠 Vector embeddings and semantic search using FAISS
- 🤖 Generative chatbot responses via Google Gemini
- 📝 Chat session history and exportable conversation logs
- 📊 PostgreSQL-backed session storage
- 🌐 Web UI served from FastAPI with Jinja2 templates

---

## 📁 Project Structure

- `app/`
  - `main.py` — FastAPI application entrypoint
  - `routes.py` — web routes, upload endpoints, chat and history pages
  - `ragpipeline.py` — query handling and LLM prompt logic
  - `vector_db.py` — FAISS vector store integration
  - `workers.py` — Celery tasks for background file processing
  - `db.py` — SQLAlchemy DB session configuration
  - `db_crud.py` — database CRUD helpers
  - `redisconfig.py` — Redis and vector DB configuration
  - `templates/` — Jinja2 HTML templates
  - `uploads/` — uploaded CSV/ZIP files and extracted contents
  - `requirements.txt` — Python dependencies
- `docker-compose.yml` — service definitions for web, Celery, PostgreSQL, and Redis
- `Dockerfile` — app container build instructions

---

## 🛠️ Tech Stack

| Component          | Tool / Library                 |
|-------------------|--------------------------------|
| Backend           | FastAPI                        |
| Web Server        | Uvicorn                        |
| Async Tasks       | Celery + Redis                 |
| Vector Store      | FAISS                          |
| Embeddings        | SentenceTransformers           |
| LLM               | Google Gemini                  |
| Database          | PostgreSQL                     |
| Templating        | Jinja2                         |
| Config Management | python-dotenv                  |

---

## ⚡ Requirements

- Python 3.11+
- Docker & Docker Compose (for containerized deployment)
- `app/requirements.txt` for Python dependencies

---

## 🚀 Getting Started

### Docker Compose (recommended)

1. From the repository root:

```bash
cd H:\fastapi
docker-compose up --build
```

2. Open the app in your browser:

```text
http://localhost:8341
```

### Local development

1. Install dependencies:

```bash
cd H:\fastapi\app
python -m pip install -r requirements.txt
```

2. Copy the example env file and update it:

```bash
cd H:\fastapi\app
copy example.env .env
```

3. Edit `app/.env` and set your values:

```env
GEMINI_API_KEY="Your Gemini API key here"
DATABASE_URL="Your database URL here"
```

4. Start FastAPI:

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

4. In another terminal, start the Celery worker:

```bash
celery -A workers worker --loglevel=info --pool=solo
```

5. Visit:

```text
http://localhost:8000
```

---

## 🧩 Usage

- Navigate to the web UI
- Upload a `.csv` or `.zip` containing legal clauses
- Ask questions using the chat interface
- View previous sessions under `/history`
- Export chat logs as CSV

---

## 📌 Important Notes

- Uploaded CSV files must include a `clause_text` column.
- ZIP uploads should contain one or more CSV files.
- The app stores embeddings under `./RAGembeddings` by default.
- Redis is used for Celery broker/backend and session state.
- `SessionMiddleware` currently uses a placeholder secret key in `app/main.py`; replace it for production.

---

## 🧪 Running with Docker Compose

The included `docker-compose.yml` defines:

- `web` service on `localhost:8341`
- `celery_worker` processing background tasks
- `db` PostgreSQL on `localhost:8342`
- `redis` on `localhost:8343`

Be sure to update `DATABASE_URL` and `REDIS_URL` if you change ports or credentials.
