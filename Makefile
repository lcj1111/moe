.PHONY: help check snapshot tree

help:
	@echo "make check     Validate project, CUDA, environments, models, and Phase 0 evidence"
	@echo "make snapshot  Save a reproducibility snapshot under artifacts/manifests"
	@echo "make tree      Show the project directory layout"
	@echo "Activate with: source env/activate.sh"

check:
	@bash -c 'source env/project.env; "$$QTOPOMOE_SGLANG_VENV/bin/python" scripts/validate_configs.py'
	@bash env/check_env.sh

snapshot:
	@bash scripts/snapshot_env.sh

tree:
	@find . -maxdepth 3 -type d -not -path './.git*' -not -path './third_party/*' | sort
