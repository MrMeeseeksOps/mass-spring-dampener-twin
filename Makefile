.PHONY: install run test lint format experiments
install:
	uv sync
run:
	uv run streamlit run src/mass_spring_twin/ui/app.py
test:
	uv run pytest -q
lint:
	uv run ruff check .
	uv run ruff format --check .
format:
	uv run ruff check --fix .
	uv run ruff format .
experiments:
	uv run python experiments/01_undamped_oscillation.py
	uv run python experiments/02_damping_regimes.py
	uv run python experiments/03_numerical_stability.py
	uv run python experiments/04_prediction_error.py
