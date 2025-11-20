#!/usr/bin/env python3
"""
Validate and optionally execute the generated ONNX binary graph.

Usage:
    python scripts/validate_onnx.py target/graph.onnx

Steps:
1) Load and check the ONNX model.
2) If onnxruntime is installed, execute zeroed inputs on available providers (CoreML/Metal/CUDA/CPU).
"""
import sys
from pathlib import Path

import onnx
from onnx import TensorProto


def try_run_inference(model_path: Path) -> None:
    try:
        import numpy as np  # type: ignore
        import onnxruntime as rt  # type: ignore
    except ImportError:
        print("onnxruntime/numpy not installed; skipping inference run.")
        return

    type_map = {
        TensorProto.FLOAT: "float32",
        TensorProto.UINT8: "uint8",
        TensorProto.INT8: "int8",
        TensorProto.INT32: "int32",
        TensorProto.FLOAT16: "float16",
        TensorProto.UINT32: "uint32",
        "tensor(float)": "float32",
        "tensor(uint8)": "uint8",
        "tensor(int8)": "int8",
        "tensor(int32)": "int32",
        "tensor(float16)": "float16",
        "tensor(uint32)": "uint32",
    }

    available = rt.get_available_providers()
    # Preferred order: CoreML (macOS), Metal, CUDA, CPU.
    order = [
        "CoreMLExecutionProvider",
        "MetalExecutionProvider",
        "CUDAExecutionProvider",
        "CPUExecutionProvider",
    ]
    ran = False
    for provider in order:
        if provider not in available:
            continue
        try:
            sess = rt.InferenceSession(model_path.as_posix(), providers=[provider])
            feeds = {}
            for inp in sess.get_inputs():
                dtype = type_map.get(inp.type)
                if dtype is None:
                    raise RuntimeError(f"Unsupported input dtype for inference: {inp.type}")
                shape = [dim if dim is not None else 1 for dim in inp.shape]
                feeds[inp.name] = np.zeros(shape, dtype=dtype)

            outputs = [o.name for o in sess.get_outputs()]
            result = sess.run(outputs, feeds)
            print(f"ONNX runtime inference succeeded on {provider}:")
            for name, arr in zip(outputs, result):
                print(f"  {name}: shape={arr.shape}, dtype={arr.dtype}")
            ran = True
        except Exception as exc:  # noqa: B902
            print(f"ONNX runtime inference failed on {provider}: {exc}")

    if not ran:
        print("No supported onnxruntime providers available for inference.")


def main() -> int:
    if len(sys.argv) != 2:
        print(__doc__)
        return 1
    model_path = Path(sys.argv[1])
    if not model_path.exists():
        raise SystemExit(f"Model file not found: {model_path}")
    model = onnx.load(model_path)
    onnx.checker.check_model(model)
    print(f"ONNX model structure is valid for {model_path}")
    try_run_inference(model_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
