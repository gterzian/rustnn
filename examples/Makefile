# Makefile for WebNN Python API Examples
#
# This Makefile provides convenient targets for running example scripts.
# Run 'make help' to see all available targets.

.PHONY: help setup venv simple matmul classify classify-cpu classify-gpu all clean

# Variables
# Use parent project's venv if available, otherwise use system Python
PARENT_VENV := ../.venv-webnn/bin/python
PYTHON := $(shell if [ -f "$(PARENT_VENV)" ]; then echo "$(PARENT_VENV)"; else echo "python3"; fi)
PIP := $(shell if [ -f "$(PARENT_VENV)" ]; then echo "../.venv-webnn/bin/pip"; else echo "pip3"; fi)
DEFAULT_IMAGE := images/test.jpg
IMAGE ?= $(DEFAULT_IMAGE)

# ONNX Runtime library path (for macOS)
ORT_LIB_DIR := ../target/onnxruntime/onnxruntime-osx-arm64-1.17.0/lib
PYTHON_WITH_ORT := $(shell if [ -d "$(ORT_LIB_DIR)" ]; then echo "DYLD_LIBRARY_PATH=$(ORT_LIB_DIR) $(PYTHON)"; else echo "$(PYTHON)"; fi)

# Default target
help:
	@echo "WebNN Python API Examples"
	@echo "========================="
	@echo ""
	@echo "Available targets:"
	@echo "  make setup              - Install Python dependencies (numpy, pillow)"
	@echo "  make simple             - Run simple example (graph building basics)"
	@echo "  make matmul             - Run matrix multiplication example"
	@echo "  make classify           - Run image classification with default image"
	@echo "  make classify IMAGE=... - Run image classification with custom image"
	@echo "  make classify-cpu       - Run classification on CPU backend"
	@echo "  make classify-gpu       - Run classification on GPU backend"
	@echo "  make all                - Run all examples"
	@echo "  make clean              - Remove generated ONNX files"
	@echo "  make help               - Show this help message"
	@echo ""
	@echo "Examples:"
	@echo "  make setup"
	@echo "  make simple"
	@echo "  make classify"
	@echo "  make classify IMAGE=~/Downloads/cat.jpg"
	@echo "  make classify-gpu"
	@echo "  make all"
	@echo ""
	@echo "Prerequisites:"
	@echo "  - Build the Python package first (from project root): make python-test"
	@echo "  - Or run: maturin develop --features python,onnx-runtime"
	@echo ""
	@echo "Default test image: $(DEFAULT_IMAGE)"
	@echo "Python: $(PYTHON)"

# Install Python dependencies
setup:
	@echo "Installing Python dependencies..."
	@$(PIP) install numpy pillow 2>/dev/null || echo "✓ Dependencies already installed"
	@echo "✓ Setup complete"

# Run simple example
simple: setup
	@echo "=========================================="
	@echo "Running Simple Example"
	@echo "=========================================="
	$(PYTHON_WITH_ORT) python_simple.py
	@echo ""

# Run matrix multiplication example
matmul: setup
	@echo "=========================================="
	@echo "Running Matrix Multiplication Example"
	@echo "=========================================="
	$(PYTHON_WITH_ORT) python_matmul.py
	@echo ""

# Run image classification example (uses default image or IMAGE parameter)
classify: setup
	@echo "=========================================="
	@echo "Running Image Classification Example"
	@echo "Image: $(IMAGE)"
	@echo "=========================================="
	$(PYTHON_WITH_ORT) image_classification.py $(IMAGE)
	@echo ""

# Run image classification on CPU backend
classify-cpu: setup
	@echo "=========================================="
	@echo "Running Image Classification (CPU Backend)"
	@echo "Image: $(IMAGE)"
	@echo "=========================================="
	$(PYTHON_WITH_ORT) image_classification.py $(IMAGE) --backend cpu
	@echo ""

# Run image classification on GPU backend
classify-gpu: setup
	@echo "=========================================="
	@echo "Running Image Classification (GPU Backend)"
	@echo "Image: $(IMAGE)"
	@echo "=========================================="
	$(PYTHON_WITH_ORT) image_classification.py $(IMAGE) --backend gpu
	@echo ""

# Run all examples
all: simple matmul classify
	@echo "=========================================="
	@echo "All examples completed successfully!"
	@echo "=========================================="

# Clean generated files
clean:
	@echo "Cleaning generated ONNX files..."
	rm -f *.onnx
	@echo "✓ Clean complete"
