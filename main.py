from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field
import httpx
import ollama
import os
import sys
from typing import Optional

app = FastAPI(title="Local LLM API", description="FastAPI with Ollama - dynamic model selection")

SELECTED_MODEL: str = ""

class ChatRequest(BaseModel):
    prompt: str
    stream: bool = False
    temperature: float = Field(default=0.7, ge=0.0, le=2.0, description="Temperature for response randomness (0.0-2.0)")
    use_search: bool = False
    search_query: Optional[str] = None
    search_max_results: int = Field(default=5, ge=1, le=10, description="Max search results to include")

class SearchRequest(BaseModel):
    query: str
    max_results: int = Field(default=5, ge=1, le=10)
    language: Optional[str] = None
    categories: Optional[list[str]] = None
    safesearch: int = Field(default=1, ge=0, le=2)

class ChatResponse(BaseModel):
    response: str

class SearchResult(BaseModel):
    title: str
    url: str
    snippet: str

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

async def searxng_search(request: SearchRequest, client_ip: str, http_request: Request) -> list[SearchResult]:
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
        results.append(
            SearchResult(
                title=item.get("title", ""),
                url=item.get("url", ""),
                snippet=item.get("content", "") or item.get("snippet", ""),
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
    return {"message": "Local LLM API is running", "model": SELECTED_MODEL}

@app.post("/search", response_model=SearchResponse)
async def search(request: SearchRequest, http_request: Request):
    """
    Run a web search via SearxNG and return results.
    """
    try:
        client_ip = get_client_ip(http_request)
        results = await searxng_search(request, client_ip, http_request)
        return SearchResponse(results=results)
    except httpx.HTTPError as e:
        raise HTTPException(status_code=502, detail=f"SearxNG error: {str(e)}")

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest, http_request: Request):
    """
    Send a prompt to the selected model and get a response
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
            # For streaming, we'll just return the full response
            # In a real streaming scenario, you'd use StreamingResponse
            full_response = ""
            for chunk in response:
                full_response += chunk['message']['content']
            return ChatResponse(response=full_response)
        else:
            return ChatResponse(response=response['message']['content'])
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error communicating with Ollama: {str(e)}")

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

if __name__ == "__main__":
    import uvicorn
    SELECTED_MODEL = select_model()
    print(f"Starting API server with model: {SELECTED_MODEL}")
    uvicorn.run(app, host="0.0.0.0", port=8000)
