# Release Notes - v0.5.2

Release Date: 2025-12-28

## Overview

v0.5.2 is a focused patch release with three commits since v0.5.1. It improves the WebNN text/JSON import pipeline, adds missing shape inference, and introduces a ready-to-run MiniLM embeddings demo from Hugging Face Hub.

## Highlights

- **WebNN text loader hardening**: Automatically sanitizes identifiers (dots/colons → underscores) and inlines weights from adjacent `manifest.json` + `model.weights` so onnx2webnn exports load without manual edits.
- **WebNN JSON shape inference**: Runs a shape inference pass during JSON import, deduplicates outputs, and exposes Python helpers (`count_unknown_shapes`, structured debug output) to spot unresolved shapes.
- **MiniLM embeddings demo**: New Hugging Face Hub demo/Make target (`make minilm-demo-hub`) with `MINILM_MODEL_ID` override plus detailed usage docs and comparison script.

## Notes

- No breaking changes. Existing graphs and Python APIs continue to work.
- Python wheel version follows Cargo version (0.5.2) via dynamic versioning.
