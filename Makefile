.PHONY: install setup lint format train train-sample train-advanced install-semantic train-semantic train-best mlflow-ui test test-unit test-integration test-all train-and-register register-model demo-inference load-model serve-model clean run-all dev workflow docker-build docker-run monitor prefect-train

install:
	pyenv install 3.11.9 || true
	pyenv local 3.11.9
	python -m venv .venv
	/bin/bash -c "source .venv/bin/activate && pip install --upgrade pip && pip install uv && uv pip install --system --strict pyproject.toml"

setup: install
	mkdir -p data/processed models/trained preprocessors

lint:
	uv run ruff check src --fix

format:
	uv run ruff format src

train:
	PYTHONPATH=src ./.venv/bin/python src/models/train.py --data data/raw/goodreads_books_children.json --config configs/model_config.yaml

train-sample:
	PYTHONPATH=src ./.venv/bin/python src/models/train.py --data data/raw/goodreads_books_children.json --config configs/model_config.yaml

train-advanced:
	PYTHONPATH=src ./.venv/bin/python src/models/advanced_train.py

install-semantic:
	uv pip install sentence-transformers torch

train-semantic:
	PYTHONPATH=src ./.venv/bin/python src/models/semantic_train.py

train-best:
	TOKENIZERS_PARALLELISM=false PYTHONPATH=src ./.venv/bin/python src/models/optimized_semantic_train.py

mlflow-ui:
	uv run mlflow ui --host 0.0.0.0 --port 5000

test:
	PYTHONPATH=src ./.venv/bin/python -m pytest tests/ -v

test-unit:
	PYTHONPATH=src ./.venv/bin/python -m pytest tests/unit/ -v

test-integration:
	PYTHONPATH=src ./.venv/bin/python -m pytest tests/integration/ -v

test-all:
	PYTHONPATH=src ./.venv/bin/python -m pytest tests/ -v

train-and-register:
	TOKENIZERS_PARALLELISM=false PYTHONPATH=src ./.venv/bin/python src/models/optimized_semantic_train.py
	PYTHONPATH=src ./.venv/bin/python src/models/model_registry.py

register-model:
	PYTHONPATH=src ./.venv/bin/python src/models/model_registry.py

demo-inference:
	PYTHONPATH=src ./.venv/bin/python src/models/model_inference.py

load-model:
	PYTHONPATH=src ./.venv/bin/python -c "from src.models.model_inference import load_latest_model; predictor = load_latest_model(); print('✅ Model loaded successfully!')"

serve-model:
	mlflow models serve -m "models:/semantic-rating-predictor/Production" -p 5001 --env-manager local

clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	rm -rf mlruns/
	rm -rf models/
	rm -rf preprocessors/
	rm -rf *.csv

run-all: setup lint train

dev: setup
	@echo "Development environment ready!"
	@echo "Run 'make train' to start training"
	@echo "Run 'make mlflow-ui' to view experiments"

docker-build:
	docker build -f Dockerfile -t review-rating-api .

docker-run:
	docker run -p 8000:8000 -v $(shell pwd)/mlruns:/mlruns review-rating-api

docker-debug:
	docker run -it -v $(shell pwd)/mlruns:/mlruns review-rating-api /bin/bash

docker-logs:
	docker logs $(shell docker ps -q --filter ancestor=review-rating-api)
monitor:
	PYTHONPATH=src ./.venv/bin/python src/models/model_monitoring.py

prefect-train:
	PYTHONPATH=src ./.venv/bin/python src/workflows/train_flow.py