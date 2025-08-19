.PHONY: install data repro serve test lint

install:
	pip install -r requirements.txt

data:
	python scripts/generate_synthetic_data.py

repro:
	dvc repro

serve:
	uvicorn app.main:app --host 0.0.0.0 --port 8000

lint:
	python -m pip install ruff==0.6.8 && ruff check .

test:
	python -m pytest -q