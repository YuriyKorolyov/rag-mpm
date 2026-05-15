import json
import logging
import os
import uuid
from collections.abc import AsyncGenerator, Iterator
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
import requests
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import BaseModel
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
import fitz

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("rag_mpm")

QDRANT_URL = os.getenv("QDRANT_URL", "http://qdrant:6333")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://ollama:11434")
NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY", "")
COLLECTION_NAME = "mpm_docs"
EMBED_MODEL = "nomic-embed-text"
OLLAMA_CHAT_MODEL = "qwen3.5:0.8b"
EMBED_DIM = 768
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100
TOP_K = 5

UPLOAD_DIR = Path("/uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

qdrant = QdrantClient(url=QDRANT_URL)


def ensure_collection() -> None:
    existing = [c.name for c in qdrant.get_collections().collections]
    if COLLECTION_NAME not in existing:
        logger.info("Creating Qdrant collection %s", COLLECTION_NAME)
        qdrant.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=EMBED_DIM, distance=Distance.COSINE),
        )
    else:
        logger.debug("Qdrant collection %s already exists", COLLECTION_NAME)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Startup: QDRANT_URL=%s OLLAMA_URL=%s", QDRANT_URL, OLLAMA_URL)
    ensure_collection()
    logger.info("Startup complete, chat model=%s", OLLAMA_CHAT_MODEL)
    yield
    logger.info("Shutdown")


app = FastAPI(
    title="RAG MPM API",
    description="RAG система для методологии 3D прогнозирования минерализации",
    version="1.0.0",
    lifespan=lifespan,
)


def extract_text_from_pdf(path: Path) -> str:
    doc = fitz.open(str(path))
    pages = []
    for page in doc:
        pages.append(page.get_text())
    doc.close()
    return "\n".join(pages)


def split_text(text: str, source: str) -> list[dict]:
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
    async with httpx.AsyncClient(timeout=60.0) as client:
        resp = await client.post(
            f"{OLLAMA_URL}/api/embeddings",
            json={"model": EMBED_MODEL, "prompt": text},
        )
        resp.raise_for_status()
        return resp.json()["embedding"]


async def search_similar(query: str, top_k: int = TOP_K) -> list[dict]:
    query_vec = await get_embedding(query)
    results = qdrant.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vec,
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
        for r in results.points
    ]


def build_prompt(query: str, chunks: list[dict]) -> str:
    context = "\n\n---\n\n".join(
        f"[Источник: {c['source']}, чанк {c['chunk_index']}]\n{c['text']}"
        for c in chunks
    )
    return (
        "Ты — научный ассистент по методологии 3D прогнозирования минерализации (MPM). "
        "Отвечай точно и по делу, опираясь только на предоставленный контекст. "
        "ОТВЕЧАЙ ТОЛЬКО НА РУССКОМ ЯЗЫКЕ.\n\n"
        f"Контекст:\n{context}\n\n"
        f"Вопрос: {query}\n\n"
        "Ответ:"
    )


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


@app.post("/upload", summary="Загрузить PDF и добавить в Qdrant")
async def upload_pdf(file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        logger.warning("Reject upload: not PDF filename=%s", file.filename)
        raise HTTPException(status_code=400, detail="Только PDF файлы")

    save_path = UPLOAD_DIR / file.filename
    content = await file.read()
    save_path.write_bytes(content)
    logger.info("Saved PDF bytes=%s path=%s", len(content), save_path)

    text = extract_text_from_pdf(save_path)
    if not text.strip():
        logger.error("No text extracted from %s", file.filename)
        raise HTTPException(status_code=422, detail="Не удалось извлечь текст из PDF")

    chunks = split_text(text, source=file.filename)
    logger.info("Split into %s chunks for %s", len(chunks), file.filename)

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

    batch_size = 100
    for i in range(0, len(points), batch_size):
        qdrant.upsert(
            collection_name=COLLECTION_NAME,
            points=points[i : i + batch_size],
        )
    logger.info("Upserted %s points to Qdrant", len(points))

    return {
        "filename": file.filename,
        "chunks_created": len(chunks),
        "message": f"Успешно загружено {len(chunks)} чанков в Qdrant",
    }


@app.post("/search", response_model=EmbedResponse, summary="Поиск через embedding (без LLM)")
async def search_embedding(req: QueryRequest):
    logger.info("Search query_len=%s top_k=%s", len(req.query), req.top_k)
    results = await search_similar(req.query, req.top_k)
    logger.info("Search returned %s hits", len(results))
    return EmbedResponse(query=req.query, results=results)


@app.post("/ask/ollama", summary="RAG с Ollama qwen3.5:0.8b")
async def ask_ollama(req: QueryRequest):
    chunks = await search_similar(req.query, req.top_k)
    if not chunks:
        logger.warning("Ask ollama: no chunks for query")
        raise HTTPException(status_code=404, detail="Ничего не найдено в базе знаний")

    prompt = build_prompt(req.query, chunks)
    sources = [{"source": c["source"], "score": round(c["score"], 4)} for c in chunks]

    async def stream_ollama() -> AsyncGenerator[str, None]:
        async with httpx.AsyncClient(timeout=120.0) as client:
            async with client.stream(
                "POST",
                f"{OLLAMA_URL}/api/generate",
                json={
                    "model": OLLAMA_CHAT_MODEL,
                    "prompt": f"\\no_think {prompt}",
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

    async def full_stream() -> AsyncGenerator[str, None]:
        yield json.dumps({"sources": sources, "model": OLLAMA_CHAT_MODEL}) + "\n"
        yield "data: "
        async for token in stream_ollama():
            yield token
        yield "\n"

    logger.info("Streaming ollama response model=%s", OLLAMA_CHAT_MODEL)
    return StreamingResponse(full_stream(), media_type="text/plain")


@app.post("/ask/nvidia", summary="RAG с NVIDIA Mistral Large")
async def ask_nvidia(req: QueryRequest):
    if not NVIDIA_API_KEY:
        logger.error("NVIDIA_API_KEY missing")
        raise HTTPException(status_code=500, detail="NVIDIA_API_KEY не задан")

    chunks = await search_similar(req.query, req.top_k)
    if not chunks:
        logger.warning("Ask nvidia: no chunks for query")
        raise HTTPException(status_code=404, detail="Ничего не найдено в базе знаний")

    prompt = build_prompt(req.query, chunks)
    sources = [{"source": c["source"], "score": round(c["score"], 4)} for c in chunks]

    def stream_nvidia() -> Iterator[str]:
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
                            # Проверка как в официальном коде
                            if "choices" in data and len(data["choices"]) > 0:
                                delta = data["choices"][0].get("delta", {})
                                if content := delta.get("content"):
                                    yield content
                        except (json.JSONDecodeError, KeyError):
                            continue

    logger.info("Streaming nvidia response")
    return StreamingResponse(stream_nvidia(), media_type="text/plain")
