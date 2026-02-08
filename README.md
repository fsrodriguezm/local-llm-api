# Local LLM API

A lightweight FastAPI server powered by Ollama with standard `/v1/*` endpoints for local AI inference.

## Features

- **Standard v1 API**: `/v1/chat/completions`, `/v1/completions`, `/v1/embeddings` endpoints
- **Runtime Model Switching**: Change models via API without restarting
- **Web Search Integration**: Optional SearxNG for grounded responses
- **Embeddings Support**: Vector generation for RAG and semantic search
- **Token Tracking**: Estimate and report token usage
- **Health Monitoring**: Check Ollama connectivity and model status
- **Auto-Generated Docs**: Built-in Swagger UI at `/docs`

## Prerequisites

- Python 3.12+
- [uv](https://github.com/astral-sh/uv) package manager
- [Ollama](https://ollama.ai/) running with at least one model installed

## Quick Start

```bash
# Install dependencies
uv sync

# Pull models
ollama pull qwen2.5-coder:3b
ollama pull nomic-embed-text  # For embeddings

# Run with default model
DEFAULT_MODEL="qwen2.5-coder:3b" uv run python main.py

# Or interactive selection
uv run python main.py
```

API available at `http://localhost:8000`

## API Endpoints

### Core Endpoints

#### `GET /v1/models`
List available models.

```bash
curl http://localhost:8000/v1/models
```

#### `POST /v1/model`
Switch active model.

```bash
curl -X POST http://localhost:8000/v1/model \
  -H "Content-Type: application/json" \
  -d '{"model": "llama3.2:3b"}'
```

#### `POST /v1/chat/completions`
Chat with multi-turn conversation support.

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": "Explain FastAPI briefly."}
    ],
    "temperature": 0.7
  }'
```

**Response:**
```json
{
  "id": "chatcmpl-abc123",
  "object": "chat.completion",
  "created": 1707264000,
  "model": "qwen2.5-coder:3b",
  "choices": [{
    "index": 0,
    "message": {
      "role": "assistant",
      "content": "FastAPI is a modern Python web framework..."
    },
    "finish_reason": "stop"
  }],
  "usage": {
    "prompt_tokens": 25,
    "completion_tokens": 15,
    "total_tokens": 40
  }
}
```

#### `POST /v1/completions`
Single-shot text completion.

```bash
curl -X POST http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Write a haiku about coding",
    "temperature": 0.8
  }'
```

#### `POST /v1/embeddings`
Generate vector embeddings.

```bash
curl -X POST http://localhost:8000/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{
    "model": "nomic-embed-text",
    "input": "Your text here"
  }'
```

#### `POST /v1/search`
Web search via SearxNG (requires setup).

```bash
curl -X POST http://localhost:8000/v1/search \
  -H "Content-Type: application/json" \
  -d '{"query": "FastAPI tutorial", "max_results": 5}'
```

### Utility Endpoints

- `GET /` - API status
- `GET /health` - Health check for monitoring

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `DEFAULT_MODEL` | Model to use at startup | None (interactive) |
| `SEARXNG_URL` | SearxNG instance URL | `http://localhost:8080` |
| `SEARXNG_API_KEY` | Optional SearxNG API key | None |

## SearxNG Setup (Optional)

```bash
# Start SearxNG
make start-searxng

# Set environment
export SEARXNG_URL=http://localhost:8080

# Use in chat (legacy endpoint)
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "What is the latest Python version?",
    "use_search": true
  }'
```

## Python Client Example

```python
import requests

response = requests.post(
    "http://localhost:8000/v1/chat/completions",
    json={
        "messages": [{"role": "user", "content": "Hello!"}],
        "temperature": 0.7
    }
)

print(response.json()["choices"][0]["message"]["content"])
```

## Interactive Documentation

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## Performance Tips

- Use smaller models: `ollama pull qwen2.5-coder:3b` or `llama3.2:1b`
- Quantized models save memory: `llama3.1:8b-instruct-q4_K_M`
- Dedicated embedding models: `nomic-embed-text`, `mxbai-embed-large`
- Adjust temperature by task:
  - Factual: `0.1-0.3`
  - Creative: `0.8-1.5`
  - Code: `0.2-0.5`

## Troubleshooting

**"No model selected"**
Set `DEFAULT_MODEL` env var or run `python main.py` interactively

**"Model not found"**
Check: `ollama list`, then: `ollama pull <model-name>`

**SearxNG errors**
Verify: `make start-searxng` and check `SEARXNG_URL`

## License

MIT License

---

**Built with:** [FastAPI](https://fastapi.tiangolo.com/) • [Ollama](https://ollama.ai/) • [SearxNG](https://github.com/searxng/searxng)
