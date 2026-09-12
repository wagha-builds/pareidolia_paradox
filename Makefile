.PHONY: setup cache calibrate folds test lint fast train robustness ensemble infer submit validate app reproduce

setup:
	python -m venv .venv
	.venv/Scripts/python -m pip install -r requirements.txt

cache:
	python -m src.dataset --mode cache

calibrate:
	python -m src.canonical --mode calibrate

folds:
	python -m src.dataset --mode folds

test:
	pytest -q

lint:
	ruff check
	ruff format --check

fast:
	python -m src.train --config $(CONFIG) --fast

train:
	python -m src.train --config $(CONFIG) --seeds "$(SEEDS)"

robustness:
	python -m src.robustness --run $(RUN)

ensemble:
	python -m src.ensemble --spec $(SPEC)

infer:
	python -m src.infer --artifact $(ARTIFACT)

submit:
	python -m src.submit --artifact $(ARTIFACT)

validate:
	python -m src.submit --validate $(FILE)

app:
	docker compose up

reproduce:
	python -m src.reproduce
