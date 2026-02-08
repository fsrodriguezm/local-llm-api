from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field
import httpx
import ollama
import os
import sys
from typing import Optional, List
import trafilatura
import asyncio

app = FastAPI(
    title="Local LLM API",
    description="OpenAI-compatible API with Ollama - dynamic model selection",
    version="1.0.0"
)

SELECTED_MODEL: str = ""

@app.on_event("startup")
def init_selected_model() -> None:
    """Initialize SELECTED_MODEL from DEFAULT_MODEL when running via uvicorn."""
    global SELECTED_MODEL

    if SELECTED_MODEL:
        return

    default_model = os.getenv("DEFAULT_MODEL", "").strip()
    if not default_model:
        return

    try:
        available = get_available_models()
        if default_model in available:
            SELECTED_MODEL = default_model
            print(f"Using model from DEFAULT_MODEL: {SELECTED_MODEL}")
        else:
            print(f"Warning: DEFAULT_MODEL '{default_model}' not found.")
            print(f"Available models: {', '.join(available)}")
    except Exception as e:
        print(f"Warning: failed to initialize DEFAULT_MODEL '{default_model}': {e}")

class ChatRequest(BaseModel):
    prompt: str
    stream: bool = False
    temperature: float = Field(default=0.7, ge=0.0, le=2.0, description="Temperature for response randomness (0.0-2.0)")
    use_search: bool = False
    search_query: Optional[str] = None
    search_max_results: int = Field(default=5, ge=1, le=10, description="Max search results to include")

class Message(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    model: Optional[str] = None
    messages: List[Message]
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    stream: bool = False
    max_tokens: Optional[int] = None

class CompletionRequest(BaseModel):
    model: Optional[str] = None
    prompt: str
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    stream: bool = False
    max_tokens: Optional[int] = None

class EmbeddingRequest(BaseModel):
    model: Optional[str] = None
    input: str | List[str]

class ModelSwitchRequest(BaseModel):
    model: str

class SearchRequest(BaseModel):
    query: str
    max_results: int = Field(default=5, ge=1, le=10)
    language: Optional[str] = None
    categories: Optional[list[str]] = None
    safesearch: int = Field(default=1, ge=0, le=2)
    extract_content: bool = Field(default=False, description="Extract full page content using Trafilatura")

class ChatResponse(BaseModel):
    response: str

class Choice(BaseModel):
    index: int
    message: Message
    finish_reason: str

class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int

class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[Choice]
    usage: Usage

class CompletionChoice(BaseModel):
    index: int
    text: str
    finish_reason: str

class CompletionResponse(BaseModel):
    id: str
    object: str = "text_completion"
    created: int
    model: str
    choices: List[CompletionChoice]
    usage: Usage

class EmbeddingData(BaseModel):
    object: str = "embedding"
    embedding: List[float]
    index: int

class EmbeddingResponse(BaseModel):
    object: str = "list"
    data: List[EmbeddingData]
    model: str
    usage: Usage

class ModelInfo(BaseModel):
    id: str
    object: str = "model"
    created: int
    owned_by: str = "ollama"

class ModelsResponse(BaseModel):
    object: str = "list"
    data: List[ModelInfo]

class SearchResult(BaseModel):
    title: str
    url: str
    snippet: str
    content: Optional[str] = None  # Full extracted content via Trafilatura

class SearchResponse(BaseModel):
    results: list[SearchResult]

def get_available_models() -> list[str]:
    """Fetch available models from Ollama."""
    try:
        result = ollama.list()
        return [model.model for model in result.models]
    except Exception as e:
        print(f"Error connecting to Ollama: {e}")
        return []

def select_model() -> str:
    """CLI prompt to select a model."""
    models = get_available_models()

    if not models:
        print("No models found. Make sure Ollama is running and has models installed.")
        print("Install a model with: ollama pull <model-name>")
        sys.exit(1)

    print("\nAvailable Ollama models:")
    print("-" * 30)
    for i, model in enumerate(models, 1):
        print(f"  {i}. {model}")
    print("-" * 30)

    while True:
        try:
            choice = input(f"Select a model (1-{len(models)}): ").strip()
            index = int(choice) - 1
            if 0 <= index < len(models):
                selected = models[index]
                print(f"Selected: {selected}\n")
                return selected
            else:
                print(f"Please enter a number between 1 and {len(models)}")
        except ValueError:
            print("Please enter a valid number")
        except KeyboardInterrupt:
            print("\nExiting...")
            sys.exit(0)

def get_current_model(requested_model: Optional[str] = None) -> str:
    """Get the model to use for the request."""
    if requested_model:
        # Verify the requested model exists
        available = get_available_models()
        if requested_model not in available:
            raise HTTPException(status_code=404, detail=f"Model '{requested_model}' not found")
        return requested_model

    if not SELECTED_MODEL:
        raise HTTPException(status_code=400, detail="No model selected. Use /v1/model to select one.")

    return SELECTED_MODEL

def estimate_tokens(text: str) -> int:
    """Rough token estimation (4 chars ≈ 1 token)."""
    return len(text) // 4

def generate_id() -> str:
    """Generate a unique ID for responses."""
    import uuid
    return f"chatcmpl-{uuid.uuid4().hex[:8]}"

def get_timestamp() -> int:
    """Get current Unix timestamp."""
    import time
    return int(time.time())

def get_searxng_url() -> str:
    return os.getenv("SEARXNG_URL", "http://localhost:8080").rstrip("/")

def get_searxng_headers() -> dict[str, str]:
    api_key = os.getenv("SEARXNG_API_KEY")
    if not api_key:
        return {}
    return {"X-API-KEY": api_key}

def searxng_debug_enabled() -> bool:
    return os.getenv("SEARXNG_DEBUG", "").lower() in {"1", "true", "yes"}

def get_client_ip(http_request: Request) -> str:
    forwarded_for = http_request.headers.get("x-forwarded-for", "")
    if forwarded_for:
        return forwarded_for.split(",")[0].strip()
    real_ip = http_request.headers.get("x-real-ip")
    if real_ip:
        return real_ip
    if http_request.client and http_request.client.host:
        return http_request.client.host
    return "127.0.0.1"

def build_forward_headers(client_ip: str, http_request: Request) -> dict[str, str]:
    headers = get_searxng_headers()
    headers["X-Forwarded-For"] = client_ip
    headers["X-Real-IP"] = client_ip
    headers["X-Forwarded-Host"] = http_request.headers.get("host", "localhost")
    headers["X-Forwarded-Proto"] = http_request.headers.get("x-forwarded-proto", "http")
    user_agent = http_request.headers.get("user-agent")
    if user_agent:
        headers["User-Agent"] = user_agent
    if searxng_debug_enabled():
        print(f"SearxNG headers: {headers}")
    return headers

async def extract_content(url: str) -> Optional[str]:
    """Extract clean text content from a URL using Trafilatura."""
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(url, follow_redirects=True)
            response.raise_for_status()
            html = response.text

        # Extract main content
        content = trafilatura.extract(
            html,
            include_comments=False,
            include_tables=True,
            no_fallback=False
        )
        return content
    except Exception as e:
        print(f"Failed to extract content from {url}: {e}")
        return None

async def searxng_search(request: SearchRequest, client_ip: str, http_request: Request, extract_full_content: bool = False) -> list[SearchResult]:
    params = {
        "q": request.query,
        "format": "json",
        "safesearch": request.safesearch,
    }
    if request.language:
        params["language"] = request.language
    if request.categories:
        params["categories"] = ",".join(request.categories)

    url = f"{get_searxng_url()}/search"
    async with httpx.AsyncClient(timeout=10.0) as client:
        response = await client.get(
            url,
            params=params,
            headers=build_forward_headers(client_ip, http_request),
        )
        response.raise_for_status()
        data = response.json()

    results = []
    for item in data.get("results", [])[: request.max_results]:
        title = item.get("title", "")
        url = item.get("url", "")
        snippet = item.get("content", "") or item.get("snippet", "")

        # Optionally extract full content
        content = None
        if extract_full_content:
            content = await extract_content(url)

        results.append(
            SearchResult(
                title=title,
                url=url,
                snippet=snippet,
                content=content
            )
        )
    return results

def build_search_context(results: list[SearchResult]) -> str:
    if not results:
        return ""
    lines = ["Sources:"]
    for idx, result in enumerate(results, 1):
        lines.append(f"{idx}) {result.title} — {result.url}")
        if result.snippet:
            lines.append(f"   {result.snippet}")
    return "\n".join(lines)

def augment_search_query(prompt: str, query: str) -> str:
    if "site:" in query.lower():
        return query

    lowered = prompt.lower()
    filters = []
    if "youtube" in lowered or "yt" in lowered or "channel" in lowered:
        filters.append("site:youtube.com")
    if "reddit" in lowered or "subreddit" in lowered:
        filters.append("site:reddit.com")
    if "twitter" in lowered or "tweet" in lowered or "x.com" in lowered:
        filters.append("site:x.com OR site:twitter.com")

    if not filters:
        return query

    filter_block = " OR ".join(filters)
    return f"{query} ({filter_block})"

@app.get("/")
async def root():
    return {"message": "Local LLM API is running", "model": SELECTED_MODEL, "version": "1.0.0"}

@app.get("/health")
async def health():
    """
    Check if the API and Ollama are working
    """
    try:
        models = ollama.list()
        return {
            "status": "healthy",
            "ollama_connected": True,
            "selected_model": SELECTED_MODEL,
            "models_available": len(models.models)
        }
    except Exception as e:
        return {"status": "unhealthy", "ollama_connected": False, "error": str(e)}

# ============================================================================
# v1 API Endpoints (OpenAI-compatible)
# ============================================================================

@app.get("/v1/models", response_model=ModelsResponse)
async def list_models():
    """
    List all available models (OpenAI-compatible endpoint).
    """
    try:
        models = get_available_models()
        timestamp = get_timestamp()

        return ModelsResponse(
            object="list",
            data=[
                ModelInfo(id=model, object="model", created=timestamp, owned_by="ollama")
                for model in models
            ]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error listing models: {str(e)}")

@app.post("/v1/model")
async def switch_model(request: ModelSwitchRequest):
    """
    Switch the active model.
    """
    global SELECTED_MODEL

    models = get_available_models()
    if request.model not in models:
        raise HTTPException(
            status_code=404,
            detail=f"Model '{request.model}' not found. Available models: {', '.join(models)}"
        )

    SELECTED_MODEL = request.model
    return {
        "switched_to": SELECTED_MODEL,
        "available_models": models
    }

@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def chat_completions(request: ChatCompletionRequest):
    """
    OpenAI-compatible chat completions endpoint.
    """
    try:
        model = get_current_model(request.model)

        # Convert messages to Ollama format
        messages = [{"role": msg.role, "content": msg.content} for msg in request.messages]

        response = ollama.chat(
            model=model,
            messages=messages,
            stream=request.stream,
            options={"temperature": request.temperature}
        )

        if request.stream:
            full_response = ""
            for chunk in response:
                full_response += chunk['message']['content']
            content = full_response
        else:
            content = response['message']['content']

        # Estimate tokens
        prompt_text = " ".join([msg.content for msg in request.messages])
        prompt_tokens = estimate_tokens(prompt_text)
        completion_tokens = estimate_tokens(content)

        return ChatCompletionResponse(
            id=generate_id(),
            object="chat.completion",
            created=get_timestamp(),
            model=model,
            choices=[
                Choice(
                    index=0,
                    message=Message(role="assistant", content=content),
                    finish_reason="stop"
                )
            ],
            usage=Usage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens
            )
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

@app.post("/v1/completions", response_model=CompletionResponse)
async def completions(request: CompletionRequest):
    """
    OpenAI-compatible text completions endpoint (non-chat format).
    """
    try:
        model = get_current_model(request.model)

        response = ollama.generate(
            model=model,
            prompt=request.prompt,
            stream=request.stream,
            options={"temperature": request.temperature}
        )

        if request.stream:
            full_response = ""
            for chunk in response:
                full_response += chunk.get('response', '')
            content = full_response
        else:
            content = response['response']

        prompt_tokens = estimate_tokens(request.prompt)
        completion_tokens = estimate_tokens(content)

        return CompletionResponse(
            id=generate_id(),
            object="text_completion",
            created=get_timestamp(),
            model=model,
            choices=[
                CompletionChoice(
                    index=0,
                    text=content,
                    finish_reason="stop"
                )
            ],
            usage=Usage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens
            )
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

@app.post("/v1/embeddings", response_model=EmbeddingResponse)
async def embeddings(request: EmbeddingRequest):
    """
    Generate embeddings for text input (OpenAI-compatible).
    """
    try:
        model = get_current_model(request.model)

        # Handle both string and list inputs
        inputs = [request.input] if isinstance(request.input, str) else request.input

        embeddings_data = []
        total_tokens = 0

        for idx, text in enumerate(inputs):
            response = ollama.embeddings(model=model, prompt=text)
            embeddings_data.append(
                EmbeddingData(
                    object="embedding",
                    embedding=response['embedding'],
                    index=idx
                )
            )
            total_tokens += estimate_tokens(text)

        return EmbeddingResponse(
            object="list",
            data=embeddings_data,
            model=model,
            usage=Usage(
                prompt_tokens=total_tokens,
                completion_tokens=0,
                total_tokens=total_tokens
            )
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

@app.post("/v1/search", response_model=SearchResponse)
async def search_v1(request: SearchRequest, http_request: Request):
    """
    Run a web search via SearxNG and return results.

    Set extract_content=true to fetch and extract full page content using Trafilatura.
    """
    try:
        client_ip = get_client_ip(http_request)
        results = await searxng_search(request, client_ip, http_request, request.extract_content)
        return SearchResponse(results=results)
    except httpx.HTTPError as e:
        raise HTTPException(status_code=502, detail=f"SearxNG error: {str(e)}")

# ============================================================================
# Legacy Endpoints (for backward compatibility)
# ============================================================================

@app.post("/search", response_model=SearchResponse)
async def search(request: SearchRequest, http_request: Request):
    """
    [DEPRECATED] Use /v1/search instead.
    """
    return await search_v1(request, http_request)

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest, http_request: Request):
    """
    [DEPRECATED] Use /v1/chat/completions instead.
    Send a prompt to the selected model and get a response.
    """
    try:
        prompt = request.prompt
        if request.use_search:
            search_request = SearchRequest(
                query=augment_search_query(
                    request.prompt,
                    request.search_query or request.prompt,
                ),
                max_results=request.search_max_results,
            )
            client_ip = get_client_ip(http_request)
            results = await searxng_search(search_request, client_ip, http_request)
            context = build_search_context(results)
            if context:
                prompt = f"{context}\n\nQuestion: {request.prompt}\nAnswer with citations like [1]."

        response = ollama.chat(
            model=SELECTED_MODEL,
            messages=[{"role": "user", "content": prompt}],
            stream=request.stream,
            options={"temperature": request.temperature}
        )

        if request.stream:
            full_response = ""
            for chunk in response:
                full_response += chunk['message']['content']
            return ChatResponse(response=full_response)
        else:
            return ChatResponse(response=response['message']['content'])

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error communicating with Ollama: {str(e)}")

if __name__ == "__main__":
    import uvicorn

    # Try environment variable first, fallback to interactive selection
    default_model = os.getenv("DEFAULT_MODEL", "").strip()

    if default_model:
        # Verify the model exists
        available = get_available_models()
        if default_model in available:
            SELECTED_MODEL = default_model
            print(f"Using model from DEFAULT_MODEL: {SELECTED_MODEL}")
        else:
            print(f"Warning: DEFAULT_MODEL '{default_model}' not found.")
            print(f"Available models: {', '.join(available)}")
            SELECTED_MODEL = select_model()
    else:
        # Interactive selection
        SELECTED_MODEL = select_model()

    print(f"\nStarting API server with model: {SELECTED_MODEL}")
    print(f"API Documentation: http://localhost:8000/docs")
    print(f"Health Check: http://localhost:8000/health")
    uvicorn.run(app, host="0.0.0.0", port=8000)
