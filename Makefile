.PHONY: install install-bio test lint format clean train-cam train-rsp train-tcd train-all fetch-data preprocess evaluate smoke-test

# ── Installation ─────────────────────────────────────────────────────────────
install:
	pip install -e ".[dev]"

install-bio:
	pip install -r requirements-bio.txt

# ── Testing ───────────────────────────────────────────────────────────────────
test:
	KMP_DUPLICATE_LIB_OK=TRUE pytest tests/ -v --tb=short

test-fast:
	KMP_DUPLICATE_LIB_OK=TRUE pytest tests/ -v --tb=short -x -q --ignore=tests/test_integration

smoke-test:
	KMP_DUPLICATE_LIB_OK=TRUE python -c "import oracle; print('oracle package OK')"
	KMP_DUPLICATE_LIB_OK=TRUE python -c "from oracle.interfaces import CAMOutput, RSPOutput, TCDOutput; print('interfaces OK')"
	KMP_DUPLICATE_LIB_OK=TRUE python -c "from oracle.models.cancer_score_mlp import CancerScoreFunction; print('CancerScoreFunction OK')"
	KMP_DUPLICATE_LIB_OK=TRUE python -c "from oracle.models.switch_predictor_gnn import SwitchPredictorGNN; print('SwitchPredictorGNN OK')"

# ── Code quality ──────────────────────────────────────────────────────────────
lint:
	ruff check oracle/ tests/ scripts/

format:
	ruff format oracle/ tests/ scripts/
	isort oracle/ tests/ scripts/

typecheck:
	mypy oracle/ --ignore-missing-imports

# ── Data pipeline ─────────────────────────────────────────────────────────────
fetch-data:
	KMP_DUPLICATE_LIB_OK=TRUE python scripts/fetch_data.py

preprocess:
	KMP_DUPLICATE_LIB_OK=TRUE python scripts/preprocess_all.py

build-benchmarks:
	KMP_DUPLICATE_LIB_OK=TRUE python scripts/build_benchmarks.py

# ── Training ──────────────────────────────────────────────────────────────────
train-cam:
	KMP_DUPLICATE_LIB_OK=TRUE python scripts/train_cam.py

train-rsp:
	KMP_DUPLICATE_LIB_OK=TRUE python scripts/train_rsp.py

train-tcd:
	KMP_DUPLICATE_LIB_OK=TRUE python scripts/train_tcd.py

train-all:
	KMP_DUPLICATE_LIB_OK=TRUE python scripts/train_all.py

# ── Inference ─────────────────────────────────────────────────────────────────
run-aml:
	KMP_DUPLICATE_LIB_OK=TRUE python scripts/run_aml_pipeline.py

evaluate:
	KMP_DUPLICATE_LIB_OK=TRUE python scripts/evaluate.py

# ── Cleanup ───────────────────────────────────────────────────────────────────
clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete
	find . -name "*.pyo" -delete
	rm -rf .pytest_cache/ .mypy_cache/ .ruff_cache/

clean-checkpoints:
	rm -rf checkpoints/*.pt checkpoints/*.ckpt

clean-outputs:
	rm -rf outputs/reports/* outputs/molecules/* outputs/figures/*
