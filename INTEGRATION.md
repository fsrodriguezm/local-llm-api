# Integration Guide

## Using with agent-lab

The local-llm-api can serve as a unified backend for the agent-lab multi-agent research system.

### Setup

1. **Start the Docker stack**:
   ```bash
   cd ~/repos/local-llm-api
   docker compose up -d
   ```

2. **Verify services are running**:
   ```bash
   curl http://localhost:8000/health  # API health
   curl http://localhost:8080         # SearxNG health
   ```

### agent-lab Configuration

Create a new search tool in `agent-lab/tools/remote_search.py`:

```python
"""Remote search via local-llm-api."""
import httpx
from typing import List, Dict, Any

class RemoteSearchTool:
    """Search tool that uses local-llm-api's /v1/search endpoint."""

    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.search_url = f"{base_url}/v1/search"

    async def search(
        self,
        query: str,
        max_results: int = 5,
        extract_content: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Execute search via remote API.

        Args:
            query: Search query
            max_results: Number of results (1-10)
            extract_content: Extract full content using Trafilatura

        Returns:
            List of search results with title, url, snippet, and optionally content
        """
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                self.search_url,
                json={
                    "query": query,
                    "max_results": max_results,
                    "extract_content": extract_content
                }
            )
            response.raise_for_status()
            data = response.json()
            return data["results"]
```

Then update `agent-lab/tools/search_factory.py`:

```python
from tools.remote_search import RemoteSearchTool

def create_search_tool():
    """Create appropriate search tool based on config."""
    search_engine = os.getenv("SEARCH_ENGINE", "tavily")

    if search_engine == "remote":
        return RemoteSearchTool()
    elif search_engine == "tavily":
        return TavilySearchTool()
    else:
        return DuckDuckGoSearchTool()
```

Set in `agent-lab/.env`:
```bash
# Use remote search (local-llm-api)
SEARCH_ENGINE=remote
USE_WEB_SEARCH=true
```

### Benefits

✅ **Centralized search infrastructure** - SearxNG + Trafilatura in one place
✅ **Clean content extraction** - Trafilatura removes ads, navigation, boilerplate
✅ **Docker isolation** - Search stack independent of Python environment
✅ **Reusable** - Same search API for multiple projects
✅ **No API keys** - Self-hosted search (no Tavily costs)

### Architecture

```
┌─────────────────┐
│   agent-lab     │
│  (Researcher)   │
└────────┬────────┘
         │ HTTP
         ▼
┌─────────────────┐      ┌──────────────┐
│ local-llm-api   │─────▶│   SearxNG    │
│  (FastAPI)      │      │  (Search)    │
└────────┬────────┘      └──────────────┘
         │
         ▼
    Trafilatura
 (Content Extract)
```

### Testing

```bash
# Test search without content extraction
curl -X POST http://localhost:8000/v1/search \
  -H "Content-Type: application/json" \
  -d '{"query": "LangGraph tutorial", "max_results": 3}'

# Test with full content extraction
curl -X POST http://localhost:8000/v1/search \
  -H "Content-Type: application/json" \
  -d '{
    "query": "LangGraph tutorial",
    "max_results": 3,
    "extract_content": true
  }' | jq '.results[0].content'
```

### Performance

- **Snippets only**: ~1-2 seconds
- **With content extraction**: ~5-10 seconds (fetches and parses pages)
- **Parallel extraction**: Content fetched concurrently for all results

### Troubleshooting

**"Connection refused"**:
```bash
docker compose ps  # Check services are running
docker compose logs api  # Check API logs
```

**"SearxNG timeout"**:
```bash
docker compose logs searxng  # Check SearxNG logs
curl http://localhost:8080/healthz  # Direct SearxNG check
```

**Content extraction fails**:
- Some sites block scraping
- Check robots.txt compliance
- Trafilatura gracefully returns None for failed extractions
