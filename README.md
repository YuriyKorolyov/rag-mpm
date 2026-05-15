# RAG MPM

RAG на FastAPI, Qdrant, Ollama и NVIDIA API для вопросов по методологии MPM.

## Стек

| Компонент | Технология |
| --- | --- |
| API | FastAPI, Uvicorn |
| Векторная БД | Qdrant |
| Эмбеддинги | Ollama `nomic-embed-text` |
| LLM локальный | Ollama `qwen3.5:0.8b` |
| LLM облачный | NVIDIA API `mistral-large-3` |
| Чанкинг | LangChain `RecursiveCharacterTextSplitter` |
| Контейнеризация | Docker Compose |

## Быстрый старт

Клонировать проект, перейти в каталог `rag-mpm`.

Создать `.env` с переменной `NVIDIA_API_KEY`.

Запуск:

```bash
docker compose up --build -d
```

При первом запуске `ollama-init` подтянет `nomic-embed-text` и `qwen3.5:0.8b`. Прогресс:

```bash
docker compose logs -f ollama-init
```

Проверка API:

```bash
curl -s -X POST http://localhost:8000/search -H "Content-Type: application/json" -d "{\"query\":\"тест\",\"top_k\":3}"
```

## Логи Docker Compose в файл

Windows (из корня репозитория):

```powershell
.\scripts\docker-compose-logs-file.ps1
```

Linux/macOS:

```bash
chmod +x scripts/docker-compose-logs-file.sh
./scripts/docker-compose-logs-file.sh
```

Строки дописываются в `logs/docker-compose.log`.

Уровень логов приложения: переменная окружения `LOG_LEVEL` (например `DEBUG`).

## Нагрузочное тестирование (k6)

Нужен установленный [k6](https://k6.io/docs/get-started/installation/).

Стек должен быть запущен, в Qdrant уже должны быть точки (хотя бы один загруженный PDF), иначе ответ `/search` будет 200 с пустым `results`, проверка `search has results` может не проходить.

```bash
k6 run loadtest/rag.k6.js
```

Другой хост:

```bash
k6 run -e BASE_URL=http://127.0.0.1:8000 loadtest/rag.k6.js
```

## Эндпоинты

### POST `/upload`

Загрузка PDF, чанкинг, эмбеддинги, запись в Qdrant.

### POST `/search`

Поиск по эмбеддингам без LLM.

### POST `/ask/ollama`

RAG со стримингом через Ollama `qwen3.5:0.8b`.

### POST `/ask/nvidia`

RAG со стримингом через NVIDIA API.

Документация OpenAPI: `http://localhost:8000/docs`

## Параметры чанкинга (`app/main.py`)

| Параметр | Значение |
| --- | --- |
| `CHUNK_SIZE` | 500 |
| `CHUNK_OVERLAP` | 100 |
| `TOP_K` | 5 |
| `EMBED_DIM` | 768 |

## Структура

```
rag-mpm/
├── app/
│   ├── main.py
│   ├── requirements.txt
│   └── Dockerfile
├── loadtest/
│   └── rag.k6.js
├── scripts/
│   ├── docker-compose-logs-file.ps1
│   └── docker-compose-logs-file.sh
├── logs/
├── uploads/
├── docker-compose.yml
├── .env
├── .gitignore
└── README.md
```

## Команды

```bash
docker compose logs -f api
docker compose restart api
docker compose down
docker compose down -v
```
