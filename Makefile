.PHONY: start stop restart status logs install health clean dev help test-search

# Default model - can be overridden: make start MODEL=llama3.2:3b
MODEL ?= llama3.1:8b-instruct-q4_K_M

# Start everything (SearxNG + API)
start:
	@echo "🚀 Starting Local LLM API..."
	@docker compose up -d
	@echo "⏳ Waiting for SearxNG to be ready..."
	@sleep 3
	@echo "🔧 Starting FastAPI server..."
	@DEFAULT_MODEL=llama3.1:8b-instruct-q4_K_M SEARXNG_URL=http://localhost:8080 nohup uv run uvicorn main:app --host 0.0.0.0 --port 8000 > api.log 2>&1 &
	@echo "✅ Services started!"
	@echo ""
	@echo "📊 API: http://localhost:8000"
	@echo "📊 Docs: http://localhost:8000/docs"
	@echo "🔍 SearxNG: http://localhost:8080"
	@echo ""
	@echo "Run 'make logs' to view logs"
	@echo "Run 'make stop' to stop all services"

# Stop everything
stop:
	@echo "🛑 Stopping services..."
	-@pkill -f "uvicorn main:app" 2>/dev/null || true
	@docker compose down
	@echo "✅ All services stopped"

# Restart everything
restart: stop start

# Show service status
status:
	@echo "📊 Service Status:"
	@echo ""
	@echo "Docker containers:"
	@docker compose ps
	@echo ""
	@echo "FastAPI process:"
	@pgrep -f "uvicorn main:app" > /dev/null && echo "✅ Running (PID: $$(pgrep -f 'uvicorn main:app'))" || echo "❌ Not running"

# View logs
logs:
	@echo "📝 API logs (last 50 lines, Ctrl+C to exit):"
	@tail -f api.log 2>/dev/null || echo "No logs yet. API may not have started."

# View SearxNG logs
logs-searxng:
	@docker compose logs -f searxng

# Check health of all services
health:
	@echo "🏥 Health Check:"
	@echo ""
	@echo -n "API: "
	@curl -sf http://localhost:8000/health | jq -r '.status' 2>/dev/null || echo "❌ Not responding"
	@echo -n "SearxNG: "
	@curl -sf http://localhost:8080/healthz > /dev/null && echo "✅ healthy" || echo "❌ unhealthy"
	@echo -n "Ollama: "
	@curl -sf http://localhost:11434/api/tags > /dev/null && echo "✅ connected" || echo "❌ not connected"

# Install dependencies
install:
	@echo "📦 Installing dependencies..."
	@uv sync
	@echo "✅ Dependencies installed"

# Clean up everything
clean: stop
	@echo "🧹 Cleaning up..."
	@docker compose down -v
	@rm -rf .venv __pycache__ *.pyc api.log
	@echo "✅ Cleanup complete"

# Development mode with auto-reload (foreground)
dev:
	@echo "🔧 Starting in development mode (auto-reload)..."
	@docker compose up -d
	@echo "⏳ Waiting for SearxNG..."
	@sleep 3
	@echo "🚀 Starting API with auto-reload..."
	@DEFAULT_MODEL=$(MODEL) SEARXNG_URL=http://localhost:8080 uv run uvicorn main:app --host 0.0.0.0 --port 8000 --reload

# Test search with content extraction
test-search:
	@echo "Testing basic search..."
	@curl -s -X POST http://localhost:8000/v1/search \
		-H "Content-Type: application/json" \
		-d '{"query": "FastAPI tutorial", "max_results": 2}' | jq '.results[0]'
	@echo "\n\nTesting with content extraction..."
	@curl -s -X POST http://localhost:8000/v1/search \
		-H "Content-Type: application/json" \
		-d '{"query": "FastAPI tutorial", "max_results": 1, "extract_content": true}' | jq '.results[0].content[0:200]'

# Help
help:
	@echo "Local LLM API - Makefile Commands"
	@echo ""
	@echo "Usage: make [target] [MODEL=model-name]"
	@echo ""
	@echo "Targets:"
	@echo "  start         - Start all services (SearxNG + API)"
	@echo "  stop          - Stop all services"
	@echo "  restart       - Restart all services"
	@echo "  status        - Show service status"
	@echo "  logs          - View API logs"
	@echo "  logs-searxng  - View SearxNG logs"
	@echo "  health        - Check health of all services"
	@echo "  dev           - Start in development mode with auto-reload"
	@echo "  test-search   - Test search functionality"
	@echo "  install       - Install Python dependencies"
	@echo "  clean         - Stop services and clean up"
	@echo "  help          - Show this help message"
	@echo ""
	@echo "Examples:"
	@echo "  make start                          # Start with default model"
	@echo "  make start MODEL=llama3.2:3b        # Start with specific model"
	@echo "  make dev                            # Start with auto-reload"
	@echo "  make health                         # Check all services"
