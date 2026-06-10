WORKER4 ?= worker4
WORKER1 ?= worker1
REMOTE_DIR ?= /srv/fpga/ft-diloco

.PHONY: lint test sync fetch

lint:
	ruff check src tests analysis chaos

test:
	pytest -q

# Push code to worker4 (data/ and experiments/ are never touched by --delete).
sync:
	rsync -az --delete \
		--exclude .git --exclude .venv --exclude __pycache__ \
		--exclude data --exclude experiments \
		./ $(WORKER4):$(REMOTE_DIR)/

# Pull run artifacts back from worker4.
fetch:
	rsync -az $(WORKER4):$(REMOTE_DIR)/experiments/ ./experiments/
