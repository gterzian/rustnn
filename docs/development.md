# Development Guide

## Prerequisites

- **Rust**: 1.70+ (install from [rustup.rs](https://rustup.rs/))
- **Python**: 3.11+ with pip
- **Maturin**: `pip install maturin`
- **Optional**: Graphviz for visualization (`brew install graphviz` on macOS)

## Building from Source

```bash
# Clone repository
git clone https://github.com/tarekziade/rustnn.git
cd rustnn

# Build Rust library
cargo build --release

# Build Python package
maturin develop --features python

# Run tests
cargo test                    # Rust tests
python -m pytest tests/       # Python tests

# Build documentation
mkdocs serve                  # Live preview at http://127.0.0.1:8000
mkdocs build                  # Build static site
```

## Running Examples

### Python Examples

```bash
# Install package first
maturin develop --features python

# Run examples
python examples/python_simple.py
python examples/python_matmul.py

# Run integration tests
python tests/test_integration.py
python tests/test_coreml_basic.py --cleanup
```

### Rust Examples

```bash
cargo run -- examples/sample_graph.json --export-dot graph.dot
```

## Testing

### Python Tests

```bash
# All tests
python -m pytest tests/

# With coverage
python -m pytest tests/ --cov=webnn --cov-report=html

# Specific test
python -m pytest tests/test_python_api.py::test_context_creation -v

# Run integration tests with cleanup
python tests/test_integration.py --cleanup
```

### Rust Tests

```bash
# All tests
cargo test

# Specific module
cargo test validator

# With output
cargo test -- --nocapture

# Integration tests only
cargo test --test '*'
```

## Feature Flags

The project uses Cargo feature flags to control optional functionality:

```bash
# Python bindings
cargo build --features python

# ONNX Runtime support
cargo build --features onnx-runtime

# CoreML Runtime support (macOS only)
cargo build --features coreml-runtime

# All features
cargo build --features python,onnx-runtime,coreml-runtime

# Python package with all features
maturin develop --features python,onnx-runtime,coreml-runtime
```

## Development Workflow

### 1. Make Changes

Edit Rust code in `src/` or Python code in `python/webnn/`.

### 2. Format Code

```bash
# Rust
cargo fmt

# Python
black python/ tests/
```

### 3. Run Tests

```bash
# Quick check (Rust only)
cargo check --features python

# Full test suite
cargo test
python -m pytest tests/
```

### 4. Build and Test Python Package

```bash
maturin develop --features python
python -m pytest tests/
```

### 5. Update Documentation

Edit files in `docs/` and preview:

```bash
mkdocs serve
```

## Debugging

### Rust

```bash
# Debug build with symbols
cargo build --features python

# Run with backtrace
RUST_BACKTRACE=1 cargo run -- examples/sample_graph.json
```

### Python

```python
import webnn
import logging

logging.basicConfig(level=logging.DEBUG)

# Your code here
```

## Common Tasks

### Add a New Operation

1. Update `graph.rs` with new operation type
2. Add validation logic in `validator.rs`
3. Implement conversion in `converters/onnx.rs` and `converters/coreml.rs`
4. Add Python binding in `src/python/graph_builder.rs`
5. Add tests in `tests/test_python_api.py`

### Add a New Backend

1. Create new file in `src/executors/your_backend.rs`
2. Add feature flag in `Cargo.toml`
3. Implement executor trait/functions
4. Add conditional compilation in `src/executors/mod.rs`
5. Wire up in `src/python/context.rs` backend selection
6. Add tests

### Update Documentation

1. Edit markdown files in `docs/`
2. Preview with `mkdocs serve`
3. Check links and formatting
4. Build with `mkdocs build`

## Troubleshooting

### Maturin Build Fails

```bash
# Update Rust
rustup update

# Clean build artifacts
cargo clean
rm -rf target/

# Rebuild
maturin develop --features python
```

### Import Errors

```bash
# Ensure you're in the right virtual environment
which python

# Reinstall
maturin develop --features python --force

# Verify installation
python -c "import webnn; print(webnn.__version__)"
```

### ONNX Runtime Issues

On macOS ARM64, ONNX Runtime prebuilt binaries may not be available. Use system library:

```bash
# Install ONNX Runtime
brew install onnxruntime

# Set environment variables
export ORT_STRATEGY=system
export ORT_LIB_LOCATION=/opt/homebrew/lib

# Build
maturin develop --features python,onnx-runtime
```

### Test Failures

```bash
# Run specific failing test with verbose output
python -m pytest tests/test_python_api.py::test_name -xvs

# Check if backend is available
python -c "import webnn; ctx = webnn.ML().create_context(); print(ctx.accelerated)"
```

## Code Style

### Rust

- Follow [Rust API Guidelines](https://rust-lang.github.io/api-guidelines/)
- Use `cargo fmt` for formatting
- Use `cargo clippy` for linting
- Write doc comments for public APIs

### Python

- Follow [PEP 8](https://pep8.org/)
- Use type hints
- Write docstrings for public APIs
- Use `black` for formatting

## Git Workflow

### Commits

```bash
# Stage changes
git add .

# Commit with descriptive message
git commit -m "Add feature X

- Detail 1
- Detail 2

🤖 Generated with [Claude Code](https://claude.com/claude-code)"

# Push
git push origin main
```

### Pre-commit Hooks

The project uses pre-commit hooks to ensure code quality:

- `cargo fmt --check` - Ensures Rust code is formatted
- Tests run automatically in CI

## CI/CD

### GitHub Actions

The project uses GitHub Actions for CI:

- `.github/workflows/ci.yml` - Main CI pipeline
  - Runs on push and pull requests
  - Tests on Linux and macOS
  - Builds Python wheels
  - Runs all tests

### Local CI Simulation

```bash
# Run the same checks as CI
cargo fmt --check
cargo clippy -- -D warnings
cargo test
python -m pytest tests/
```

## Resources

- [Rust Book](https://doc.rust-lang.org/book/)
- [PyO3 Guide](https://pyo3.rs/)
- [W3C WebNN Spec](https://www.w3.org/TR/webnn/)
- [ONNX Documentation](https://onnx.ai/)
- [CoreML Documentation](https://developer.apple.com/documentation/coreml)
