.PHONY: start-searxng stop-searxng

SEARXNG_COMPOSE ?= searxng.compose.yml

start-searxng:
	docker compose -f $(SEARXNG_COMPOSE) up -d

stop-searxng:
	docker compose -f $(SEARXNG_COMPOSE) down
