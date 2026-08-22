.PHONY: help check snapshot tree

help:
	@echo "make check     检查项目配置、CUDA、隔离环境、模型与 Phase 0 证据"
	@echo "make snapshot  将复现环境快照保存到 artifacts/manifests"
	@echo "make tree      显示三级目录结构"
	@echo "使用前执行：source env/activate.sh"

check:
	@bash -c 'source env/project.env; "$$QTOPOMOE_SGLANG_VENV/bin/python" scripts/validate_configs.py'
	@bash env/check_env.sh

snapshot:
	@bash scripts/snapshot_env.sh

tree:
	@find . -maxdepth 3 -type d -not -path './.git*' -not -path './third_party/*' | sort
