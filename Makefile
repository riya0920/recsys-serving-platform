.PHONY: install test train ranker bench serve chaos mlflow
install:
	pip install -r requirements.txt
test:
	pytest
train:
	PYTHONPATH=src python -m recsys.train --epochs 3
bench:
	PYTHONPATH=src python -m recsys.bench_ann --n-items 50000 --queries 1000
ranker:
	PYTHONPATH=src python -m recsys.train_ranker --candidates 200
serve:
	PYTHONPATH=src uvicorn service.app:app --host 0.0.0.0 --port 8000
chaos:
	PYTHONPATH=src python -m service.chaos --url http://localhost:8000 --duration 18 --concurrency 32
mlflow:
	mlflow ui --backend-store-uri sqlite:///artifacts/mlflow.db
