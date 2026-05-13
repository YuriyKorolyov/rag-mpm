import os
import uuid
import httpx
import json
import requests
from pathlib import Path
from typing import AsyncGenerator

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
import fitz  # PyMuPDF

# ─── Config ───────────────────────────────────────────────────────────────────
QDRANT_URL      = os.getenv("QDRANT_URL", "http://qdrant:6333")
OLLAMA_URL      = os.getenv("OLLAMA_URL", "http://ollama:11434")
NVIDIA_API_KEY  = os.getenv("NVIDIA_API_KEY", "")
COLLECTION_NAME = "mpm_docs"
EMBED_MODEL     = "nomic-embed-text"
EMBED_DIM       = 768
CHUNK_SIZE      = 500
CHUNK_OVERLAP   = 100
TOP_K           = 5

UPLOAD_DIR = Path("/uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

# ─── FastAPI ───────────────────────────────────────────────────────────────────
app = FastAPI(
    title="RAG MPM API",
    description="RAG система для методологии 3D прогнозирования минерализации",
    version="1.0.0",
)

# ─── Qdrant client ────────────────────────────────────────────────────────────
qdrant = QdrantClient(url=QDRANT_URL)


def ensure_collection():
    """Создать коллекцию в Qdrant если не существует."""
    existing = [c.name for c in qdrant.get_collections().collections]
    if COLLECTION_NAME not in existing:
        qdrant.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=EMBED_DIM, distance=Distance.COSINE),
        )


# ─── Helpers ──────────────────────────────────────────────────────────────────
def extract_text_from_pdf(path: Path) -> str:
    """Извлечь текст из PDF через PyMuPDF."""
    doc = fitz.open(str(path))
    pages = []
    for page in doc:
        pages.append(page.get_text())
    doc.close()
    return "\n".join(pages)


def split_text(text: str, source: str) -> list[dict]:
    """Разбить текст на чанки с метаданными."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_text(text)
    return [
        {"text": chunk, "source": source, "chunk_index": i}
        for i, chunk in enumerate(chunks)
    ]


async def get_embedding(text: str) -> list[float]:
    """Получить эмбеддинг через Ollama nomic-embed-text."""
    async with httpx.AsyncClient(timeout=60.0) as client:
        resp = await client.post(
            f"{OLLAMA_URL}/api/embeddings",
            json={"model": EMBED_MODEL, "prompt": text},
        )
        resp.raise_for_status()
        return resp.json()["embedding"]


async def search_similar(query: str, top_k: int = TOP_K) -> list[dict]:
    """Найти похожие чанки в Qdrant."""
    query_vec = await get_embedding(query)
    results = qdrant.search(
        collection_name=COLLECTION_NAME,
        query_vector=query_vec,
        limit=top_k,
        with_payload=True,
    )
    return [
        {
            "text": r.payload["text"],
            "source": r.payload["source"],
            "chunk_index": r.payload["chunk_index"],
            "score": r.score,
        }
        for r in results
    ]


def build_prompt(query: str, chunks: list[dict]) -> str:
    """Собрать промпт из запроса и найденных чанков."""
    context = "\n\n---\n\n".join(
        f"[Источник: {c['source']}, чанк {c['chunk_index']}]\n{c['text']}"
        for c in chunks
    )
    return (
        "Ты — научный ассистент по методологии 3D прогнозирования минерализации (MPM). "
        "Отвечай точно и по делу, опираясь только на предоставленный контекст.\n\n"
        f"Контекст:\n{context}\n\n"
        f"Вопрос: {query}\n\n"
        "Ответ:"
    )


# ─── Pydantic schemas ─────────────────────────────────────────────────────────
class QueryRequest(BaseModel):
    query: str
    top_k: int = TOP_K


class SearchResult(BaseModel):
    text: str
    source: str
    chunk_index: int
    score: float


class EmbedResponse(BaseModel):
    query: str
    results: list[SearchResult]


# ─── Startup ──────────────────────────────────────────────────────────────────
@app.on_event("startup")
async def startup():
    ensure_collection()


# ─── Endpoint 1: Загрузка PDF ─────────────────────────────────────────────────
@app.post("/upload", summary="1. Загрузить PDF и добавить в Qdrant")
async def upload_pdf(file: UploadFile = File(...)):
    """
    Загружает PDF, извлекает текст, разбивает на чанки,
    создаёт эмбеддинги через Ollama и сохраняет в Qdrant.
    """
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Только PDF файлы")

    # Сохранить файл
    save_path = UPLOAD_DIR / file.filename
    content = await file.read()
    save_path.write_bytes(content)

    # Извлечь текст
    text = extract_text_from_pdf(save_path)
    if not text.strip():
        raise HTTPException(status_code=422, detail="Не удалось извлечь текст из PDF")

    # Разбить на чанки
    chunks = split_text(text, source=file.filename)

    # Создать эмбеддинги и загрузить в Qdrant
    points = []
    for chunk in chunks:
        embedding = await get_embedding(chunk["text"])
        points.append(
            PointStruct(
                id=str(uuid.uuid4()),
                vector=embedding,
                payload=chunk,
            )
        )

    # Загружаем батчами по 100
    batch_size = 100
    for i in range(0, len(points), batch_size):
        qdrant.upsert(
            collection_name=COLLECTION_NAME,
            points=points[i : i + batch_size],
        )

    return {
        "filename": file.filename,
        "chunks_created": len(chunks),
        "message": f"Успешно загружено {len(chunks)} чанков в Qdrant",
    }


# ─── Endpoint 2: Поиск только эмбеддинги ─────────────────────────────────────
@app.post("/search", response_model=EmbedResponse, summary="2. Поиск через embedding (без LLM)")
async def search_embedding(req: QueryRequest):
    """
    Ищет релевантные чанки в Qdrant по косинусному сходству.
    Возвращает top_k наиболее похожих фрагментов без генерации ответа.
    """
    results = await search_similar(req.query, req.top_k)
    return EmbedResponse(query=req.query, results=results)


# ─── Endpoint 3: Embedding + Ollama Qwen ─────────────────────────────────────
@app.post("/ask/ollama", summary="3. RAG с Ollama qwen3:0.6b")
async def ask_ollama(req: QueryRequest):
    """
    Поиск релевантных чанков + генерация ответа через Ollama (qwen3:0.6b).
    Возвращает стриминг ответа.
    """
    chunks = await search_similar(req.query, req.top_k)
    if not chunks:
        raise HTTPException(status_code=404, detail="Ничего не найдено в базе знаний")

    prompt = build_prompt(req.query, chunks)

    async def stream_ollama() -> AsyncGenerator[str, None]:
        async with httpx.AsyncClient(timeout=120.0) as client:
            async with client.stream(
                "POST",
                f"{OLLAMA_URL}/api/generate",
                json={
                    "model": "qwen3:0.6b",
                    "prompt": prompt,
                    "stream": True,
                    "options": {"temperature": 0.15, "top_p": 1.0},
                },
            ) as resp:
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    if line:
                        try:
                            data = json.loads(line)
                            if token := data.get("response"):
                                yield token
                            if data.get("done"):
                                break
                        except json.JSONDecodeError:
                            continue

    # Метаданные источников
    sources = [{"source": c["source"], "score": round(c["score"], 4)} for c in chunks]

    async def full_stream():
        # Сначала отдаём метаданные
        yield json.dumps({"sources": sources, "model": "qwen3:0.6b"}) + "\n"
        yield "data: "
        async for token in stream_ollama():
            yield token
        yield "\n"

    return StreamingResponse(full_stream(), media_type="text/plain")


# ─── Endpoint 4: Embedding + NVIDIA Mistral ───────────────────────────────────
@app.post("/ask/nvidia", summary="4. RAG с NVIDIA Mistral Large")
async def ask_nvidia(req: QueryRequest):
    """
    Поиск релевантных чанков + генерация через NVIDIA API
    (mistralai/mistral-large-3-675b-instruct-2512) со стримингом.
    """
    if not NVIDIA_API_KEY:
        raise HTTPException(status_code=500, detail="NVIDIA_API_KEY не задан")

    chunks = await search_similar(req.query, req.top_k)
    if not chunks:
        raise HTTPException(status_code=404, detail="Ничего не найдено в базе знаний")

    prompt = build_prompt(req.query, chunks)
    sources = [{"source": c["source"], "score": round(c["score"], 4)} for c in chunks]

    def stream_nvidia() -> AsyncGenerator[str, None]:
        headers = {
            "Authorization": f"Bearer {NVIDIA_API_KEY}",
            "Accept": "text/event-stream",
            "Content-Type": "application/json",
        }
        payload = {
            "model": "mistralai/mistral-large-3-675b-instruct-2512",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 2048,
            "temperature": 0.15,
            "top_p": 1.0,
            "frequency_penalty": 0.0,
            "presence_penalty": 0.0,
            "stream": True,
        }

        # Сначала метаданные
        yield json.dumps({"sources": sources, "model": "mistral-large-3"}) + "\n"

        with requests.post(
            "https://integrate.api.nvidia.com/v1/chat/completions",
            headers=headers,
            json=payload,
            stream=True,
            timeout=120,
        ) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if line:
                    decoded = line.decode("utf-8")
                    if decoded.startswith("data: "):
                        data_str = decoded[6:]
                        if data_str.strip() == "[DONE]":
                            break
                        try:
                            data = json.loads(data_str)
                            delta = data["choices"][0]["delta"]
                            if content := delta.get("content"):
                                yield content
                        except (json.JSONDecodeError, KeyError):
                            continue

    return StreamingResponse(stream_nvidia(), media_type="text/plain")


# ─── Health ───────────────────────────────────────────────────────────────────
@app.get("/health", summary="Проверка состояния сервисов")
async def health():
    status = {"api": "ok", "qdrant": "unknown", "ollama": "unknown"}

    try:
        qdrant.get_collections()
        status["qdrant"] = "ok"
    except Exception as e:
        status["qdrant"] = f"error: {e}"

    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            r = await client.get(f"{OLLAMA_URL}/api/tags")
            models = [m["name"] for m in r.json().get("models", [])]
            status["ollama"] = "ok"
            status["ollama_models"] = models
    except Exception as e:
        status["ollama"] = f"error: {e}"

    return status


@app.get("/collections", summary="Статистика коллекции Qdrant")
async def collections_info():
    info = qdrant.get_collection(COLLECTION_NAME)
    return {
        "collection": COLLECTION_NAME,
        "vectors_count": info.vectors_count,
        "points_count": info.points_count,
        "status": info.status,
    }
