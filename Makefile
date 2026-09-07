# 作用：汇总环境检查、快照和目录查看等常用维护命令。
.PHONY: help check test check-gpu snapshot tree
PYTHON ?= python

help:
	@echo "make check     检查配置、文档路径与发布清单（无需 GPU）"
	@echo "make test      CPU 测试（checkpoint 依赖见 README）"
	@echo "make check-gpu 检查 CUDA、隔离环境、模型与 Phase 0 证据"
	@echo "make snapshot  将复现环境快照保存到 artifacts/manifests"
	@echo "make tree      显示三级目录结构"
	@echo "CPU 检查依赖见 README；GPU 命令前执行 source env/activate.sh"

check:
	@$(PYTHON) scripts/validate_configs.py
	@$(PYTHON) scripts/check_document_references.py
	@$(PYTHON) scripts/check_release_manifest.py

test:
	@$(PYTHON) -m pytest --ignore=tests/test_canonicalize_qwen35_checkpoint.py -q

check-gpu:
	@bash env/check_env.sh

snapshot:
	@bash scripts/snapshot_env.sh

tree:
	@find . -maxdepth 3 -type d -not -path './.git*' -not -path './third_party/*' | sort
