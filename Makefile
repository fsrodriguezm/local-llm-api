.PHONY: up down logs restart build clean health test-search

# Start all services (API + SearxNG)
up:
	docker compose up -d

# Stop all services
down:
	docker compose down

# View logs
logs:
	docker compose logs -f

# Restart services
restart:
	docker compose restart

# Rebuild and start
build:
	docker compose up -d --build

# Clean everything (including volumes)
clean:
	docker compose down -v

# Check health
health:
	@echo "Checking API health..."
	@curl -s http://localhost:8000/health | jq .
	@echo "\nChecking SearxNG health..."
	@curl -s http://localhost:8080/healthz || echo "SearxNG not ready"

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

# Legacy commands for backward compatibility
.PHONY: start-searxng stop-searxng
SEARXNG_COMPOSE ?= searxng.compose.yml

start-searxng:
	docker compose -f $(SEARXNG_COMPOSE) up -d

stop-searxng:
	docker compose -f $(SEARXNG_COMPOSE) down
