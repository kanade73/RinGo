.DEFAULT_GOAL := help
SHELL := /bin/bash
.PHONY: help setup check lint typecheck test dev lock

help: ## Show targets
	@grep -hE '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | sed -e 's/:.*## / — /'

setup: ## Resolve Swift package dependencies from the lockfile
	swift package resolve

check: lint typecheck test ## All quality gates

lint: ## Format-check + lint
	swiftformat --lint .
	swiftlint lint --strict

typecheck: ## Compile without running tests
	swift build

test: ## Unit tests
	swift test

dev: ## Run locally
	@echo "No runnable target yet — see .claude/docs/design.md"

lock: ## Regenerate Package.resolved (sanctioned dependency-update path)
	swift package resolve

oracle: ## Build reference KataGo (Eigen backend) into .tools/ for parity goldens
	mkdir -p .tools/katago-eigen-build
	KG="$(abspath $(if $(KATAGO_REF),$(if $(filter /%,$(KATAGO_REF)),$(KATAGO_REF),$(CURDIR)/$(KATAGO_REF)),$(CURDIR)/../katago-origin/KataGo))"; cd .tools/katago-eigen-build && cmake "$$KG/cpp" -DUSE_BACKEND=EIGEN -DCMAKE_BUILD_TYPE=Release && make -j8
	@echo "oracle at .tools/katago-eigen-build/katago"

goldens: ## Regenerate parity golden fixtures using the oracle
	Scripts/gen-feature-goldens.sh
	Scripts/gen-nn-goldens.sh
