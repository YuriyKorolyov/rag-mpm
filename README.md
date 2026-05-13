# RAG MPM — система для методологии 3D прогнозирования минерализации

RAG-система на основе FastAPI + Qdrant + Ollama + NVIDIA API.  
Отвечает на вопросы по методологии MPM (BLSMOTE, Random Forest, Return-Risk и др.)  
на основе загруженных научных статей.

---

## Стек

| Компонент | Технология |
|---|---|
| API | FastAPI + Uvicorn |
| Векторная БД | Qdrant |
| Эмбеддинги | Ollama `nomic-embed-text` |
| LLM локальный | Ollama `qwen3:0.6b` |
| LLM облачный | NVIDIA API `mistral-large-3` |
| Чанкинг | LangChain `RecursiveCharacterTextSplitter` |
| Контейнеризация | Docker Compose |

---

## Быстрый старт

### 1. Клонировать / распаковать проект

```bash
cd rag-mpm
```

### 2. Заполнить .env

```bash
cp .env .env.local   # опционально
```

Отредактировать `.env`:
```
NVIDIA_API_KEY=nvapi-xxxxxxxxxxxxxxxxxxxxxx
```

### 3. Запустить

```bash
docker compose up --build -d
```

При первом запуске `ollama-init` автоматически скачает:
- `nomic-embed-text` (~270 MB) — модель эмбеддингов
- `qwen3:0.6b` (~500 MB) — локальная LLM

Дождитесь завершения (проверить логи):
```bash
docker compose logs -f ollama-init
```

### 4. Проверить статус

```bash
curl http://localhost:8000/health
```

---

## Эндпоинты

### POST `/upload` — загрузить PDF

```bash
curl -X POST http://localhost:8000/upload \
  -F "file=@minerals-13-01384.pdf"
```

**Ответ:**
```json
{
  "filename": "minerals-13-01384.pdf",
  "chunks_created": 87,
  "message": "Успешно загружено 87 чанков в Qdrant"
}
```

---

### POST `/search` — поиск только эмбеддинги (без LLM)

```bash
curl -X POST http://localhost:8000/search \
  -H "Content-Type: application/json" \
  -d '{"query": "Как работает Borderline-SMOTE?", "top_k": 5}'
```

**Ответ:** top-K наиболее релевантных чанков с оценкой сходства.

---

### POST `/ask/ollama` — RAG с локальной LLM (qwen3:0.6b)

```bash
curl -X POST http://localhost:8000/ask/ollama \
  -H "Content-Type: application/json" \
  -d '{"query": "Объясни алгоритм Random Forest для задачи MPM"}' \
  --no-buffer
```

**Ответ:** стриминг — сначала JSON с источниками, затем текст ответа.

---

### POST `/ask/nvidia` — RAG с NVIDIA Mistral Large

```bash
curl -X POST http://localhost:8000/ask/nvidia \
  -H "Content-Type: application/json" \
  -d '{"query": "Что такое анализ доходность-риск в контексте MPM?"}' \
  --no-buffer
```

**Ответ:** стриминг через NVIDIA API.

---

### GET `/health` — состояние сервисов

```bash
curl http://localhost:8000/health
```

### GET `/collections` — статистика коллекции Qdrant

```bash
curl http://localhost:8000/collections
```

---

## Swagger UI

Полная интерактивная документация:  
`http://localhost:8000/docs`

---

## Параметры чанкинга (настраиваются в `app/main.py`)

| Параметр | Значение | Описание |
|---|---|---|
| `CHUNK_SIZE` | 500 | Размер чанка в символах |
| `CHUNK_OVERLAP` | 100 | Перекрытие между чанками |
| `TOP_K` | 5 | Кол-во чанков для контекста |
| `EMBED_DIM` | 768 | Размерность nomic-embed-text |

---

## Рекомендуемые документы для загрузки

1. `minerals-13-01384.pdf` — статья Ланнигоу (главный источник)
2. Оригинальная статья SMOTE (Chawla et al., 2002) — arxiv.org/abs/1106.1813
3. Borderline-SMOTE (Han et al., 2005)
4. Random Forest (Breiman, 2001)
5. Статьи по MPM из MDPI Minerals

---

## Структура проекта

```
rag-mpm/
├── app/
│   ├── main.py           # FastAPI приложение
│   ├── requirements.txt  # Python зависимости
│   └── Dockerfile
├── uploads/              # Загруженные PDF (создаётся автоматически)
├── docker-compose.yml
├── .env                  # NVIDIA_API_KEY
├── .gitignore
└── README.md
```

---

## Полезные команды

```bash
# Посмотреть логи API
docker compose logs -f api

# Перезапустить только API (после изменений кода)
docker compose restart api

# Остановить всё
docker compose down

# Удалить данные Qdrant (сброс базы)
docker compose down -v
```
