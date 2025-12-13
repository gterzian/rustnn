"""
Shared pytest fixtures for all test files.

This module provides common fixtures for WebNN API testing.
"""

import pytest
import numpy as np

# Try to import webnn module
try:
    import webnn
    WEBNN_AVAILABLE = True
except ImportError:
    WEBNN_AVAILABLE = False


def _has_onnx_runtime():
    """Check if ONNX runtime is available for actual computation"""
    if not WEBNN_AVAILABLE:
        return False
    try:
        ml = webnn.ML()
        ctx = ml.create_context(power_preference="default", accelerated=False)
        builder = ctx.create_graph_builder()
        x = builder.input("x", [1, 1], "float32")
        y = builder.relu(x)
        graph = builder.build({"output": y})
        result = ctx.compute(graph, {"x": np.array([[1.0]], dtype=np.float32)})
        # If ONNX runtime is available, result should be non-zero
        return np.any(result["output"] != 0)
    except:
        return False


def _has_coreml_runtime():
    """Check if CoreML runtime is available (macOS only)"""
    if not WEBNN_AVAILABLE:
        return False
    try:
        import platform
        if platform.system() != "Darwin":
            return False
        ml = webnn.ML()
        # CoreML is selected when accelerated=True on macOS
        ctx = ml.create_context(power_preference="default", accelerated=True)
        # Check if CoreML methods are available
        return hasattr(ctx, 'execute_with_coreml')
    except:
        return False


ONNX_RUNTIME_AVAILABLE = _has_onnx_runtime()
COREML_RUNTIME_AVAILABLE = _has_coreml_runtime()


# Pytest markers
def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers", "wpt: WebNN W3C Web Platform Tests"
    )
    config.addinivalue_line(
        "markers", "requires_onnx_runtime: Test requires ONNX Runtime"
    )
    config.addinivalue_line(
        "markers", "requires_coreml_runtime: Test requires CoreML Runtime"
    )


@pytest.fixture(scope="session")
def ml():
    """Create ML instance (session-scoped)."""
    if not WEBNN_AVAILABLE:
        pytest.skip("webnn not built yet")
    return webnn.ML()


@pytest.fixture(params=[
    pytest.param(("onnx", False), id="onnx") if ONNX_RUNTIME_AVAILABLE else pytest.param((None, None), id="no_onnx", marks=pytest.mark.skip),
    pytest.param(("coreml", True), id="coreml") if COREML_RUNTIME_AVAILABLE else pytest.param((None, None), id="no_coreml", marks=pytest.mark.skip),
])
def context(request, ml):
    """Create ML context for the specified backend."""
    backend_name, accelerated = request.param

    if backend_name is None:
        pytest.skip("Backend not available")

    ctx = ml.create_context(power_preference="default", accelerated=accelerated)
    return ctx


@pytest.fixture
def builder(context):
    """Create graph builder."""
    return context.create_graph_builder()


# Mark for tests requiring ONNX runtime
requires_onnx_runtime = pytest.mark.skipif(
    not ONNX_RUNTIME_AVAILABLE,
    reason="ONNX runtime not available - built without onnx-runtime feature"
)

# Mark for tests requiring CoreML runtime
requires_coreml_runtime = pytest.mark.skipif(
    not COREML_RUNTIME_AVAILABLE,
    reason="CoreML runtime not available - macOS only or built without coreml-runtime feature"
)
