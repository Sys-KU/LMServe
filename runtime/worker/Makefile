PROTO_DIR        := ../../proto
PY_PROTO_OUT_DIR := llm_worker/pb
PROTOS           := worker.proto

PROTO_FILES      := $(addprefix $(PROTO_DIR)/, $(PROTOS))

PB_FILES         := $(addprefix $(PY_PROTO_OUT_DIR)/, $(PROTOS:.proto=_pb2.py))
PB_GRPC_FILES    := $(addprefix $(PY_PROTO_OUT_DIR)/, $(PROTOS:.proto=_pb2_grpc.py))

UV 				 := $(shell command -v uv 2>/dev/null)
PYTHON           := python3

ifeq ($(UV),)
    $(info 'uv' not found - falling back to pip.)
    $(info Hint: Install uv with 'curl -Ls https://astral.sh/uv/install.sh | sh'.)
    PIP_CMD := pip
else
    $(info Found 'uv' at $(UV); using it for installs.)
    PIP_CMD := $(UV) pip
    PYTHON  := $(UV) run python
endif

.PHONY: install build-dev compile-pb clean check-grpc-tools

install: compile-pb
	@echo "Installing llm-worker and Python dependencies..."
	@$(PIP_CMD) install .

build-dev: compile-pb
	@echo "Installing llm-worker and Python dependencies (editable)..."
	@$(PIP_CMD) install -e .

compile-pb: check-grpc-tools $(PB_FILES) $(PB_GRPC_FILES)
	@touch $(PY_PROTO_OUT_DIR)/__init__.py

$(PY_PROTO_OUT_DIR)/%_pb2.py: $(PROTO_DIR)/%.proto
	@mkdir -p $(PY_PROTO_OUT_DIR)
	$(PYTHON) -m grpc_tools.protoc \
		--proto_path=$(PROTO_DIR) \
		--python_out=$(PY_PROTO_OUT_DIR) \
		$<

$(PY_PROTO_OUT_DIR)/%_pb2_grpc.py: $(PROTO_DIR)/%.proto
	@mkdir -p $(PY_PROTO_OUT_DIR)
	$(PYTHON) -m grpc_tools.protoc \
		--proto_path=$(PROTO_DIR) \
		--grpc_python_out=$(PY_PROTO_OUT_DIR) \
		$<
	find $(PY_PROTO_OUT_DIR) -type f -name "*_pb2_grpc.py" -exec sed -i -e 's/^\(import.*pb2\)/from . \1/g' {} \;

check-grpc-tools:
	@python3 -c "import grpc_tools" 2>/dev/null || \
		( echo "Installing grpcio-tools..."; $(PIP_CMD) install grpcio-tools )

clean:
	rm -f $(PY_PROTO_OUT_DIR)/*.py
