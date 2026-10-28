from __future__ import annotations

import json
import os
import re
import sqlite3
import uuid
from contextlib import contextmanager
from collections import Counter
from pathlib import Path
from typing import Any

import httpx
import numpy as np
from docx import Document
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pypdf import PdfReader
from pydantic import BaseModel, Field
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

BASE_DIR = Path(__file__).resolve().parent
DB_PATH = Path(os.environ.get("STUDYSPACE_DB", BASE_DIR / "studyspace.db"))
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434").rstrip("/")
CHAT_MODEL = os.environ.get("OLLAMA_CHAT_MODEL", "llama3.2")
EMBED_MODEL = os.environ.get("OLLAMA_EMBED_MODEL", "nomic-embed-text")
MAX_UPLOAD = 20 * 1024 * 1024

app = FastAPI(title="Studyspace API", version="1.0.0", description="Course-grounded study tools with retrieval, citations, quizzes, and learner progress.")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=False, allow_methods=["*"], allow_headers=["*"])


@contextmanager
def db():
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA foreign_keys = ON")
    try:
        yield con
        con.commit()
    except Exception:
        con.rollback()
        raise
    finally:
        con.close()


def init_db() -> None:
    with db() as con:
        con.executescript("""
        CREATE TABLE IF NOT EXISTS courses (
          id TEXT PRIMARY KEY, name TEXT NOT NULL, term TEXT NOT NULL DEFAULT 'My course', created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE IF NOT EXISTS materials (
          id TEXT PRIMARY KEY, course_id TEXT NOT NULL REFERENCES courses(id) ON DELETE CASCADE,
          filename TEXT NOT NULL, title TEXT NOT NULL, file_type TEXT NOT NULL, pages INTEGER NOT NULL DEFAULT 0,
          chunk_count INTEGER NOT NULL DEFAULT 0, created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE IF NOT EXISTS chunks (
          id TEXT PRIMARY KEY, material_id TEXT NOT NULL REFERENCES materials(id) ON DELETE CASCADE,
          course_id TEXT NOT NULL REFERENCES courses(id) ON DELETE CASCADE, text TEXT NOT NULL,
          page INTEGER, embedding TEXT, topic TEXT NOT NULL DEFAULT 'Course material'
        );
        CREATE TABLE IF NOT EXISTS attempts (
          id TEXT PRIMARY KEY, course_id TEXT NOT NULL REFERENCES courses(id) ON DELETE CASCADE,
          topic TEXT NOT NULL, correct INTEGER NOT NULL, total INTEGER NOT NULL, created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE IF NOT EXISTS topic_progress (
          course_id TEXT NOT NULL REFERENCES courses(id) ON DELETE CASCADE, topic TEXT NOT NULL,
          correct INTEGER NOT NULL DEFAULT 0, total INTEGER NOT NULL DEFAULT 0,
          PRIMARY KEY(course_id, topic)
        );
        """)


init_db()


class CourseIn(BaseModel):
    name: str = Field(min_length=1, max_length=100)
    term: str = Field(default="My course", max_length=60)


class ChatIn(BaseModel):
    course_id: str
    question: str = Field(min_length=2, max_length=1200)


class QuizIn(BaseModel):
    course_id: str
    topic: str = Field(min_length=1, max_length=120)
    count: int = Field(default=5, ge=3, le=8)


class QuizSubmitIn(BaseModel):
    course_id: str
    topic: str
    answers: list[dict[str, Any]]


def ollama_embed(texts: list[str]) -> list[list[float]] | None:
    try:
        response = httpx.post(f"{OLLAMA_URL}/api/embed", json={"model": EMBED_MODEL, "input": texts}, timeout=3)
        if response.status_code == 404:
            # Ollama versions before /api/embed accept one input at a time.
            vectors = []
            for text in texts:
                old = httpx.post(f"{OLLAMA_URL}/api/embeddings", json={"model": EMBED_MODEL, "prompt": text}, timeout=3)
                old.raise_for_status()
                vectors.append(old.json()["embedding"])
            return vectors
        response.raise_for_status()
        data = response.json()
        return data.get("embeddings")
    except (httpx.HTTPError, ValueError, KeyError):
        return None


def ollama_chat(messages: list[dict[str, str]], json_mode: bool = False) -> str | None:
    try:
        payload = {"model": CHAT_MODEL, "messages": messages, "stream": False, "options": {"temperature": 0.25}}
        if json_mode:
            payload["format"] = "json"
        response = httpx.post(f"{OLLAMA_URL}/api/chat", json=payload, timeout=45)
        response.raise_for_status()
        return response.json().get("message", {}).get("content")
    except (httpx.HTTPError, ValueError, KeyError):
        return None


def infer_topic(text: str, title: str = "") -> str:
    words = re.findall(r"[A-Za-z][A-Za-z'-]{3,}", f"{title} {text}")
    stop = {"this", "that", "with", "from", "which", "their", "there", "these", "those", "when", "where", "what", "into", "also", "because", "while", "using", "each", "than", "then", "most", "some", "such", "have", "been", "were", "will", "about", "after", "before", "between", "through", "course", "lecture", "chapter", "slide", "slides"}
    counts = Counter(w.lower() for w in words if w.lower() not in stop)
    if not counts:
        return "Course material"
    # Prefer meaningful title words; otherwise use the most frequent content terms.
    title_words = [w.lower() for w in re.findall(r"[A-Za-z][A-Za-z'-]{3,}", title) if w.lower() not in stop]
    picked = []
    for word in title_words + [w for w, _ in counts.most_common(5)]:
        if word not in picked:
            picked.append(word)
        if len(picked) == 2:
            break
    return " ".join(w.capitalize() for w in picked) or "Course material"


def chunk_pages(pages: list[tuple[int | None, str]], size: int = 1050, overlap: int = 160) -> list[tuple[int | None, str]]:
    out: list[tuple[int | None, str]] = []
    for page, body in pages:
        text = re.sub(r"\s+", " ", body).strip()
        if not text:
            continue
        start = 0
        while start < len(text):
            end = min(len(text), start + size)
            if end < len(text):
                boundary = text.rfind(". ", start + size // 2, end)
                if boundary > start:
                    end = boundary + 1
            piece = text[start:end].strip()
            if len(piece) > 45:
                out.append((page, piece))
            if end >= len(text):
                break
            start = max(start + 1, end - overlap)
    return out


def extract_file(filename: str, raw: bytes) -> tuple[list[tuple[int | None, str]], int]:
    suffix = Path(filename).suffix.lower()
    if suffix == ".pdf":
        try:
            reader = PdfReader(__import__("io").BytesIO(raw))
            pages = [(i + 1, p.extract_text() or "") for i, p in enumerate(reader.pages)]
            return pages, len(reader.pages)
        except Exception as exc:
            raise HTTPException(422, f"Could not read this PDF: {exc}")
    if suffix == ".docx":
        try:
            document = Document(__import__("io").BytesIO(raw))
            return [(None, "\n".join(p.text for p in document.paragraphs if p.text.strip()))], 0
        except Exception as exc:
            raise HTTPException(422, f"Could not read this Word document: {exc}")
    if suffix in {".txt", ".md", ".markdown"}:
        return [(None, raw.decode("utf-8", errors="replace"))], 0
    raise HTTPException(415, "Supported formats: PDF, DOCX, TXT, and Markdown.")


def material_rows(course_id: str) -> list[sqlite3.Row]:
    with db() as con:
        return con.execute("SELECT * FROM chunks WHERE course_id=?", (course_id,)).fetchall()


def retrieve(course_id: str, query: str, limit: int = 5) -> list[dict[str, Any]]:
    chunks = material_rows(course_id)
    if not chunks:
        return []
    texts = [r["text"] for r in chunks]
    word = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), sublinear_tf=True)
    char = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=1, sublinear_tf=True)
    try:
        word_matrix = word.fit_transform(texts + [query])
        scores = cosine_similarity(word_matrix[-1], word_matrix[:-1]).ravel()
    except ValueError:
        scores = np.zeros(len(texts))
    try:
        char_matrix = char.fit_transform(texts + [query])
        char_scores = cosine_similarity(char_matrix[-1], char_matrix[:-1]).ravel()
        scores = 0.78 * scores + 0.22 * char_scores
    except ValueError:
        pass
    query_emb = ollama_embed([query])
    if query_emb and query_emb[0]:
        vectors, indices = [], []
        for i, row in enumerate(chunks):
            try:
                val = json.loads(row["embedding"] or "null")
                if val and len(val) == len(query_emb[0]):
                    vectors.append(val)
                    indices.append(i)
            except (ValueError, TypeError):
                pass
        if vectors:
            sem = cosine_similarity(np.asarray(query_emb, dtype=float), np.asarray(vectors, dtype=float)).ravel()
            semantic_scores = np.zeros(len(texts))
            semantic_scores[indices] = sem
            scores = 0.48 * scores + 0.52 * semantic_scores
    order = np.argsort(scores)[::-1][:limit]
    result = []
    for rank, i in enumerate(order):
        row = chunks[int(i)]
        result.append({"id": row["id"], "text": row["text"], "page": row["page"], "topic": row["topic"], "material_id": row["material_id"], "score": round(float(scores[i]), 4), "rank": rank + 1})
    return result


def source_label(item: dict[str, Any], index: int) -> dict[str, Any]:
    with db() as con:
        m = con.execute("SELECT title, filename FROM materials WHERE id=?", (item["material_id"],)).fetchone()
    title = m["title"] if m else "Course material"
    location = f"p. {item['page']}" if item["page"] else "document"
    return {"id": f"S{index + 1}", "material_id": item["material_id"], "title": title, "location": location, "text": item["text"], "score": item["score"]}


def course_exists(course_id: str) -> None:
    with db() as con:
        if not con.execute("SELECT 1 FROM courses WHERE id=?", (course_id,)).fetchone():
            raise HTTPException(404, "Course not found.")


@app.get("/api/health")
def health() -> dict[str, Any]:
    chat = False
    embed = False
    try:
        tags = httpx.get(f"{OLLAMA_URL}/api/tags", timeout=2).json().get("models", [])
        names = [m.get("name", "") for m in tags]
        chat = any(CHAT_MODEL in n for n in names)
        embed = any(EMBED_MODEL in n for n in names)
    except httpx.HTTPError:
        pass
    return {"status": "ok", "chat_model_available": chat, "embedding_model_available": embed, "chat_model": CHAT_MODEL, "embedding_model": EMBED_MODEL}


@app.get("/api/dashboard")
def dashboard() -> dict[str, Any]:
    with db() as con:
        courses = [dict(r) for r in con.execute("SELECT * FROM courses ORDER BY created_at")]
        result = []
        for course in courses:
            mats = con.execute("SELECT id,title,filename,file_type,pages,chunk_count,created_at FROM materials WHERE course_id=? ORDER BY created_at DESC", (course["id"],)).fetchall()
            topics = con.execute("SELECT topic,correct,total FROM topic_progress WHERE course_id=? ORDER BY total DESC, topic", (course["id"],)).fetchall()
            result.append({**course, "materials": [dict(m) for m in mats], "topics": [{"name": t["topic"], "accuracy": round(100 * (t["correct"] + 1) / (t["total"] + 2)), "attempts": t["total"]} for t in topics]})
        count = con.execute("SELECT COALESCE(SUM(total),0) FROM topic_progress").fetchone()[0]
    return {"courses": result, "attempts": count}


@app.post("/api/courses")
def create_course(payload: CourseIn) -> dict[str, Any]:
    course_id = str(uuid.uuid4())
    with db() as con:
        con.execute("INSERT INTO courses(id,name,term) VALUES(?,?,?)", (course_id, payload.name.strip(), payload.term.strip()))
    return {"id": course_id, "name": payload.name.strip(), "term": payload.term.strip(), "materials": [], "topics": []}


@app.get("/api/courses/{course_id}/materials")
def list_materials(course_id: str) -> list[dict[str, Any]]:
    course_exists(course_id)
    with db() as con:
        return [dict(r) for r in con.execute("SELECT id,title,filename,file_type,pages,chunk_count,created_at FROM materials WHERE course_id=? ORDER BY created_at DESC", (course_id,))]


@app.post("/api/courses/{course_id}/materials")
async def upload_material(course_id: str, file: UploadFile = File(...)) -> dict[str, Any]:
    course_exists(course_id)
    raw = await file.read(MAX_UPLOAD + 1)
    if len(raw) > MAX_UPLOAD:
        raise HTTPException(413, "Files must be smaller than 20 MB.")
    filename = Path(file.filename or "course-file.txt").name
    pages, page_count = extract_file(filename, raw)
    pieces = chunk_pages(pages)
    if not pieces:
        raise HTTPException(422, "No selectable text was found. Scanned PDFs need OCR before they can be indexed.")
    mid = str(uuid.uuid4())
    title = Path(filename).stem.replace("_", " ").replace("-", " ").strip().title()
    rows = []
    for page, text in pieces:
        rows.append({"id": str(uuid.uuid4()), "page": page, "text": text, "topic": infer_topic(text, title)})
    vectors = ollama_embed([r["text"] for r in rows])
    with db() as con:
        con.execute("INSERT INTO materials(id,course_id,filename,title,file_type,pages,chunk_count) VALUES(?,?,?,?,?,?,?)", (mid, course_id, filename, title, Path(filename).suffix.lower().lstrip("."), page_count, len(rows)))
        for i, row in enumerate(rows):
            vector = json.dumps(vectors[i]) if vectors and i < len(vectors) else None
            con.execute("INSERT INTO chunks(id,material_id,course_id,text,page,embedding,topic) VALUES(?,?,?,?,?,?,?)", (row["id"], mid, course_id, row["text"], row["page"], vector, row["topic"]))
    return {"id": mid, "title": title, "filename": filename, "file_type": Path(filename).suffix.lower().lstrip("."), "pages": page_count, "chunk_count": len(rows), "semantic_embeddings": bool(vectors)}


@app.delete("/api/materials/{material_id}")
def delete_material(material_id: str) -> dict[str, bool]:
    with db() as con:
        cur = con.execute("DELETE FROM materials WHERE id=?", (material_id,))
        if not cur.rowcount:
            raise HTTPException(404, "Material not found.")
    return {"deleted": True}


@app.post("/api/chat")
def chat(payload: ChatIn) -> dict[str, Any]:
    course_exists(payload.course_id)
    evidence = retrieve(payload.course_id, payload.question, limit=5)
    if not evidence:
        return {"answer": "There are no indexed course materials for this course yet. Add a PDF, Word document, or text file to start asking questions.", "sources": [], "mode": "no-materials", "confidence": 0.0}
    sources = [source_label(item, i) for i, item in enumerate(evidence)]
    context = "\n\n".join(f"[{sources[i]['id']}] {sources[i]['title']} ({sources[i]['location']}): {item['text']}" for i, item in enumerate(evidence))
    prompt = f"Answer the student's question using only the provided course passages. Be clear, concise, and helpful. Cite claims inline using the exact source IDs like [S1]. If evidence is insufficient, say what is missing rather than guessing.\n\nQuestion: {payload.question}\n\nCourse passages:\n{context}"
    generated = ollama_chat([{"role": "system", "content": "You are a careful study companion. Use only retrieved course evidence and cite it. Do not complete graded work for the student; explain concepts and reasoning."}, {"role": "user", "content": prompt}])
    if generated:
        answer, mode = generated, "ollama"
    else:
        best = evidence[0]
        answer = f"I found a relevant passage in {sources[0]['title']} ({sources[0]['location']}):\n\n“{best['text']}”\n\nThat passage is the strongest match in your materials. Start with this source, and ask a more focused follow-up if you want help unpacking a specific part."
        mode = "extractive"
    return {"answer": answer, "sources": sources, "mode": mode, "confidence": round(float(evidence[0]["score"]), 3)}


def fallback_questions(topic: str, evidence: list[dict[str, Any]], count: int) -> list[dict[str, Any]]:
    source = evidence[0]["text"] if evidence else "Review the key ideas and relationships in your course materials."
    source_obj = source_label(evidence[0], 0) if evidence else None
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", source) if len(s.strip().split()) >= 8]
    questions = []
    templates = [
        (f"Which statement best summarizes a key idea about {topic}?", "Choose the answer that matches the central idea in the cited passage."),
        (f"What relationship does your material describe for {topic}?", "Focus on the cause, role, or sequence described in the course source."),
        (f"Which detail about {topic} is supported by the source?", "Use the cited source to distinguish the supported detail from the alternatives."),
    ]
    for i in range(count):
        sentence = sentences[i % len(sentences)] if sentences else source
        correct_text = sentence[:180].rstrip(" ,;:")
        alternatives = [
            f"The source says the opposite: {correct_text[:105]}",
            f"The source does not connect this idea to {topic}.",
            "The course material says there is not enough evidence to describe this concept.",
        ]
        options = [correct_text] + alternatives
        shift = i % 4
        options = options[shift:] + options[:shift]
        item = evidence[i % len(evidence)] if evidence else None
        support = source_label(item, i % len(evidence)) if item else source_obj
        questions.append({"id": str(uuid.uuid4()), "question": templates[i % len(templates)][0], "options": options, "correct_index": (4 - shift) % 4, "explanation": f"The cited passage states: {correct_text}", "source": support})
    return questions


@app.post("/api/quizzes/generate")
def generate_quiz(payload: QuizIn) -> dict[str, Any]:
    course_exists(payload.course_id)
    evidence = retrieve(payload.course_id, payload.topic, limit=8)
    if not evidence:
        raise HTTPException(404, "Add course materials before generating practice questions.")
    context = "\n\n".join(f"[S{i+1}] {item['text']}" for i, item in enumerate(evidence))
    content = ollama_chat([
        {"role": "system", "content": "Create useful educational multiple-choice questions grounded only in the supplied passages. Return JSON with a questions array. Each question has question, four options, correct_index (0-3), explanation, and source_id copied from the passage label such as S1. Test understanding, not trivia. Do not invent facts."},
        {"role": "user", "content": f"Create {payload.count} distinct questions about {payload.topic}. Cite the passage that directly supports each question and answer. Passages:\n{context}"}
    ], json_mode=True)
    questions = None
    if content:
        try:
            parsed = json.loads(content)
            raw_questions = parsed.get("questions", [])
            valid = []
            for item in raw_questions:
                opts = item.get("options", [])
                ci = int(item.get("correct_index", -1))
                if item.get("question") and len(opts) == 4 and 0 <= ci < 4:
                    source_index = max(0, min(len(evidence) - 1, int(str(item.get("source_id", "S1")).lstrip("S")) - 1))
                    valid.append({"id": str(uuid.uuid4()), "question": str(item["question"]), "options": [str(x) for x in opts], "correct_index": ci, "explanation": str(item.get("explanation", "Review the cited passage.")), "source": source_label(evidence[source_index], source_index)})
            if valid:
                questions = valid[:payload.count]
        except (ValueError, TypeError, AttributeError):
            questions = None
    mode = "ollama" if questions else "retrieval-fallback"
    if not questions:
        questions = fallback_questions(payload.topic, evidence, payload.count)
    return {"topic": payload.topic, "questions": questions, "mode": mode, "note": "Questions are draft practice material. Check the source citation and adjust difficulty to suit your course."}


@app.post("/api/quizzes/submit")
def submit_quiz(payload: QuizSubmitIn) -> dict[str, Any]:
    course_exists(payload.course_id)
    total = len(payload.answers)
    if not total:
        raise HTTPException(400, "Submit at least one answered question.")
    correct = sum(1 for item in payload.answers if item.get("correct") is True)
    topic = payload.topic.strip() or "Course material"
    with db() as con:
        con.execute("INSERT INTO attempts(id,course_id,topic,correct,total) VALUES(?,?,?,?,?)", (str(uuid.uuid4()), payload.course_id, topic, correct, total))
        con.execute("INSERT INTO topic_progress(course_id,topic,correct,total) VALUES(?,?,?,?) ON CONFLICT(course_id,topic) DO UPDATE SET correct=correct+excluded.correct,total=total+excluded.total", (payload.course_id, topic, correct, total))
        row = con.execute("SELECT correct,total FROM topic_progress WHERE course_id=? AND topic=?", (payload.course_id, topic)).fetchone()
    accuracy = round(100 * (row["correct"] + 1) / (row["total"] + 2))
    return {"correct": correct, "total": total, "accuracy": accuracy, "topic": topic}


@app.post("/api/demo")
def load_demo() -> dict[str, Any]:
    with db() as con:
        existing = con.execute("SELECT id FROM courses WHERE name='Introduction to Biology' LIMIT 1").fetchone()
        if existing:
            course_id = existing["id"]
            if con.execute("SELECT 1 FROM materials WHERE course_id=? LIMIT 1", (course_id,)).fetchone():
                return {"course_id": course_id, "already_loaded": True}
        else:
            course_id = str(uuid.uuid4())
            con.execute("INSERT INTO courses(id,name,term) VALUES(?,?,?)", (course_id, "Introduction to Biology", "Sample course"))
    sample_docs = [
        ("Cell Biology Lecture 04", """Cellular respiration is the process by which cells harvest energy from glucose and transfer it to ATP. Glycolysis takes place in the cytoplasm and splits one glucose molecule into two pyruvate molecules. This first stage produces a net gain of two ATP and two NADH. In eukaryotic cells, pyruvate enters the mitochondrion for further oxidation. The Krebs cycle takes place in the mitochondrial matrix and produces carbon dioxide, NADH, FADH2, and a small amount of ATP. The electron transport chain is located in the inner mitochondrial membrane. Electrons from NADH and FADH2 move through a series of protein complexes. The released energy pumps protons into the intermembrane space, creating an electrochemical gradient. ATP synthase uses the flow of protons back into the matrix to make ATP. Oxygen acts as the final electron acceptor at the end of the electron transport chain. Oxygen combines with electrons and protons to form water. Without oxygen, electron flow stops and oxidative phosphorylation cannot continue."""),
        ("Course Textbook Chapter 7", """Aerobic respiration can yield around thirty ATP molecules per glucose, though the exact number varies with cell type and conditions. Fermentation allows glycolysis to continue when oxygen is unavailable by regenerating NAD+. Fermentation does not use the electron transport chain and produces no additional ATP beyond the two ATP from glycolysis. Lactic acid fermentation occurs in some animal cells and bacteria. Alcoholic fermentation in yeast produces ethanol and carbon dioxide. Photosynthesis captures light energy and stores it in chemical bonds in glucose. Cellular respiration releases chemical energy from glucose to produce ATP. The products of photosynthesis, glucose and oxygen, are reactants in cellular respiration. Carbon dioxide and water from respiration can be used as reactants in photosynthesis. These processes connect energy flow and matter cycling in ecosystems."""),
        ("Photosynthesis Seminar", """Photosynthesis occurs in chloroplasts. The light-dependent reactions take place in the thylakoid membranes, where pigments absorb light energy. Water molecules are split, releasing oxygen, electrons, and protons. The electron transport chain builds a proton gradient that powers ATP synthase. The Calvin cycle takes place in the stroma and uses ATP and NADPH to fix carbon dioxide into sugar molecules. Chlorophyll absorbs mostly red and blue wavelengths and reflects much of the green light. Light intensity, carbon dioxide concentration, and temperature can limit the rate of photosynthesis. Photosynthesis and cellular respiration are complementary but are not simply exact reversals: each has distinct chemical steps and cellular locations."""),
    ]
    inserted = 0
    for title, body in sample_docs:
        with db() as con:
            if con.execute("SELECT 1 FROM materials WHERE course_id=? AND title=?", (course_id, title)).fetchone():
                continue
        mid = str(uuid.uuid4())
        pages = [(None, body)]
        pieces = chunk_pages(pages, size=650, overlap=60)
        vectors = ollama_embed([p[1] for p in pieces])
        with db() as con:
            con.execute("INSERT INTO materials(id,course_id,filename,title,file_type,pages,chunk_count) VALUES(?,?,?,?,?,?,?)", (mid, course_id, f"{title}.txt", title, "txt", 0, len(pieces)))
            for i, (page, text) in enumerate(pieces):
                vector = json.dumps(vectors[i]) if vectors and i < len(vectors) else None
                con.execute("INSERT INTO chunks(id,material_id,course_id,text,page,embedding,topic) VALUES(?,?,?,?,?,?,?)", (str(uuid.uuid4()), mid, course_id, text, page, vector, infer_topic(text, title)))
        inserted += 1
    return {"course_id": course_id, "materials_added": inserted, "already_loaded": inserted == 0}
