.PHONY: install test train bench serve fmt
install:
	pip install -r requirements.txt
test:
	pytest
train:
	PYTHONPATH=src python -m recsys.train --epochs 3
bench:
	PYTHONPATH=src python -m recsys.bench_ann --n-items 50000 --queries 1000
serve:
	PYTHONPATH=src uvicorn service.app:app --host 0.0.0.0 --port 8000
