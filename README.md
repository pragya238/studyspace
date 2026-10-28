# Studyspace

A student-first personalized learning workspace. Studyspace turns course files into a searchable, source-grounded study space. Students decide what to review, ask questions in their own words, and practice concepts with progress saved locally.

## Features

- Upload PDF, DOCX, Markdown, and text files; extract text and split it into overlapping chunks.
- Retrieve relevant passages with TF-IDF lexical search, optionally enhanced with locally generated Ollama embeddings.
- Ask questions and get answers with page-level source citations. When Ollama is available, answers are generated from retrieved course passages; without it, the API returns a transparent extractive response.
- Generate topic quizzes from retrieved passages, submit answers, and track performance by topic.
- Persist courses, materials, chunks, practice attempts, and topic mastery in SQLite.
- Responsive student dashboard using a black, indigo, and lime visual theme.

## Requirements

- Python 3.10+
- Node.js 20+
- Optional: [Ollama](https://ollama.com/) for local LLM answers, quiz generation, and semantic embeddings.

## Run locally

Start the API:

```bash
cd backend
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
uvicorn app:app --reload --port 8000
```

In another terminal, start the web app:

```bash
cd frontend
npm install
npm run dev
```

Open http://localhost:5173. The frontend calls the API at http://localhost:8000. Set `VITE_API_URL` if the API is hosted elsewhere.

To enable local model generation, install and start Ollama, then pull models:

```bash
ollama pull llama3.2
ollama pull nomic-embed-text
```

Set optional environment variables before starting the API:

```bash
OLLAMA_URL=http://localhost:11434
OLLAMA_CHAT_MODEL=llama3.2
OLLAMA_EMBED_MODEL=nomic-embed-text
```

Without Ollama, file ingestion, lexical retrieval, citations, extractive answers, quiz practice, and progress tracking remain available. The UI indicates when a response uses the local model or retrieval fallback.

## API overview

- `GET /api/health` — API and local model availability
- `GET /api/dashboard` — course overview and topic mastery
- `POST /api/courses` — create a course
- `POST /api/courses/{course_id}/materials` — upload a supported document
- `POST /api/demo` — load a small sample Biology course
- `GET /api/courses/{course_id}/materials` — list course materials
- `DELETE /api/materials/{material_id}` — remove a material and its indexed chunks
- `POST /api/chat` — retrieve evidence and answer a course question
- `POST /api/quizzes/generate` — make a quiz from retrieved course material
- `POST /api/quizzes/submit` — save a practice attempt and update topic mastery

Interactive API documentation is available at http://localhost:8000/docs.

## RAG and learner model

Each document is extracted with its page metadata, divided into overlapping chunks, and stored in SQLite. Retrieval combines a per-query TF-IDF cosine score with cosine similarity over Ollama embeddings when configured. The top passages are passed to the model with an instruction to use only supplied evidence. Every retrieved passage is returned as a citation, and the UI shows the source label and page. Quiz submissions update a per-course, per-topic smoothed accuracy estimate; this is a study signal, not a claim of validated mastery.

## Limitations

This is a local-first educational prototype, not an assessment authority. PDF extraction works for text PDFs and does not OCR scanned pages. Local model quality and speed depend on the device and chosen Ollama models. The fallback quiz generator is intentionally simple; inspect generated questions against the cited material.

## Retrieval benchmark

After starting the backend, run the included small smoke benchmark:

```bash
cd backend
python evaluate_retrieval.py
```

It loads the sample materials if needed and reports hit-rate@5 and MRR@5 for a labeled set of course questions. Treat this as a starter evaluation set; expand it with queries and relevant passages from your own course corpus before using metrics to compare retrieval changes.
