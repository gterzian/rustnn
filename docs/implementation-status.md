# WebNN Implementation Status & Testing Strategy

**Last Updated:** 2025-12-13

## Executive Summary

rustnn implements 85 of ~95 WebNN operations (89% coverage) with full backend support across ONNX Runtime, CoreML MLProgram, and TensorRT.

**Current Status:**
- ✓ 85 operations fully implemented (Shape Inference + Python API + ONNX + CoreML)
- ✓ WPT test infrastructure in place
- ✓ WPT test data converter working (Node.js-based extraction)
- ⚠ WPT test data population incomplete (1/54 conformance files converted)
- ⚠ 260 Python API tests exist but skipped (runtime dependencies)

---

## Implementation Status

**Legend:**
- ✓ = Fully implemented
- ⚠ = Partially implemented
- ✗ = Not implemented
- ⏭ = Intentionally deferred

### All Operations (Alphabetically Sorted)

| Operation | Shape | Python | ONNX | CoreML | WPT |
|-----------|:-----:|:------:|:----:|:------:|:---:|
| `abs` | ✓ | ✓ | ✓ | ✓ | ⚠ |
| `acos` | ✓ | ✓ | ✓ | ✓ | - |
| `acosh` | ✓ | ✓ | ✓ | ✓ | - |
| `add` | ✓ | ✓ | ✓ | ✓ | ⚠ |
| `argMax` | ✓ | ✓ | ✓ | ✓ | - |
| `argMin` | ✓ | ✓ | ✓ | ✓ | - |
| `asin` | ✓ | ✓ | ✓ | ✓ | - |
| `asinh` | ✓ | ✓ | ✓ | ✓ | - |
| `atan` | ✓ | ✓ | ✓ | ✓ | - |
| `atanh` | ✓ | ✓ | ✓ | ✓ | - |
| `average_pool2d` | ✓ | ✓ | ✓ | ✓ | - |
| `batch_normalization` | ✓ | ✓ | ✓ | ✓ | ⚠ |
| `cast` | ✓ | ✓ | ✓ | ✓ | ⚠ |
| `ceil` | ✓ | ✓ | ✓ | ✓ | ⚠ |
| `concat` | ✓ | ✓ | ✓ | ✓ | ⚠ |
| `conv2d` | ✓ | ✓ | ✓ | ✓ | ⚠ |
| `conv_transpose2d` | ✓ | ✓ | ✓ | ✓ | ⚠ |
| `cos` | ✓ | ✓ | ✓ | ✓ | - |
| `cosh` | ✓ | ✓ | ✓ | ✓ | - |
| `dequantize_linear` | ✓ | ✓ | ✓ | ✓ | - |
| `div` | ✓ | ✓ | ✓ | ✓ | ⚠ |
| `elu` | ✓ | ✓ | ✓ | ✓ | ⚠ |
| `equal` | ✓ | ✓ | ✓ | ✓ | ⚠ |
| `erf` | ✓ | ✓ | ✓ | ✓ | - |
| `exp` | ✓ | ✓ | ✓ | ✓ | ⚠ |
| `expand` | ✓ | ✓ | ✓ | ✓ | ⚠ |
| `floor` | ✓ | ✓ | ✓ | ✓ | ⚠ |
| `gather` | ✓ | ✓ | ✓ | ✓ | ⚠ |
| `gelu` | ✓ | ✓ | ✓ | ✓ | - |
| `global_average_pool` | ✓ | ✓ | ✓ | ✓ | - |
| `global_max_pool` | ✓ | ✓ | ✓ | ✓ | - |
| `greater` | ✓ | ✓ | ✓ | ✓ | ⚠ |
| `greater_or_equal` | ✓ | ✓ | ✓ | ✓ | ⚠ |
| `gru` | ⏭ | ⏭ | ⏭ | ⏭ | - |
| `gruCell` | ⏭ | ⏭ | ⏭ | ⏭ | - |
| `hardSigmoid` | ✓ | ✓ | ✓ | ✓ | ⚠ |
| `hardSwish` | ✓ | ✓ | ✓ | ✓ | ⚠ |
| `identity` | ✓ | ✓ | ✓ | ✓ | - |
| `instance_normalization` | ✓ | ✓ | ✓ | ✓ | ⚠ |
| `layer_normalization` | ✓ | ✓ | ✓ | ✓ | ⚠ |
| `leakyRelu` | ✓ | ✓ | ✓ | ✓ | ⚠ |
| `lesser` | ✓ | ✓ | ✓ | ✓ | ⚠ |
| `lesser_or_equal` | ✓ | ✓ | ✓ | ✓ | ⚠ |
| `log` | ✓ | ✓ | ✓ | ✓ | ⚠ |
| `logical_and` | ✓ | ✓ | ✓ | ✓ | - |
| `logical_not` | ✓ | ✓ | ✓ | ✓ | ⚠ |
| `logical_or` | ✓ | ✓ | ✓ | ✓ | - |
| `logical_xor` | ✓ | ✓ | ✓ | ✓ | - |
| `lstm` | ⏭ | ⏭ | ⏭ | ⏭ | - |
| `lstmCell` | ⏭ | ⏭ | ⏭ | ⏭ | - |
| `matmul` | ✓ | ✓ | ✓ | ✓ | ⚠ |
| `max_pool2d` | ✓ | ✓ | ✓ | ✓ | - |
| `mul` | ✓ | ✓ | ✓ | ✓ | ⚠ |
| `neg` | ✓ | ✓ | ✓ | ✓ | ⚠ |
| `pad` | ✓ | ✓ | ✓ | ✓ | - |
| `pow` | ✓ | ✓ | ✓ | ✓ | ⚠ |
| `prelu` | ✓ | ✓ | ✓ | ✓ | - |
| `quantize_linear` | ✓ | ✓ | ✓ | ✓ | - |
| `reciprocal` | ✓ | ✓ | ✓ | ✓ | - |
| `reduce_l1` | ✓ | ✓ | ✓ | ✓ | ⚠ |
| `reduce_l2` | ✓ | ✓ | ✓ | ✓ | ⚠ |
| `reduce_log_sum` | ✓ | ✓ | ✓ | ✓ | ⚠ |
| `reduce_log_sum_exp` | ✓ | ✓ | ✓ | ✓ | ⚠ |
| `reduce_max` | ✓ | ✓ | ✓ | ✓ | ⚠ |
| `reduce_mean` | ✓ | ✓ | ✓ | ✓ | ⚠ |
| `reduce_min` | ✓ | ✓ | ✓ | ✓ | ⚠ |
| `reduce_product` | ✓ | ✓ | ✓ | ✓ | ⚠ |
| `reduce_sum` | ✓ | ✓ | ✓ | ✓ | ⚠ |
| `reduce_sum_square` | ✓ | ✓ | ✓ | ✓ | ⚠ |
| `relu` | ✓ | ✓ | ✓ | ✓ | ✓ |
| `reshape` | ✓ | ✓ | ✓ | ✓ | ⚠ |
| `round` | ✓ | ✓ | ✓ | ✓ | - |
| `scatterElements` | ✓ | ✓ | ✓ | ✓ | - |
| `scatterND` | ✓ | ✓ | ✓ | ✓ | - |
| `sigmoid` | ✓ | ✓ | ✓ | ✓ | ⚠ |
| `sign` | ✓ | ✓ | ✓ | ✓ | - |
| `sin` | ✓ | ✓ | ✓ | ✓ | - |
| `sinh` | ✓ | ✓ | ✓ | ✓ | - |
| `slice` | ✓ | ✓ | ✓ | ✓ | ⚠ |
| `softmax` | ✓ | ✓ | ✓ | ✓ | ⚠ |
| `softplus` | ✓ | ✓ | ✓ | ✓ | - |
| `softsign` | ✓ | ✓ | ✓ | ✓ | - |
| `split` | ✓ | ✓ | ✓ | ✓ | ⚠ |
| `sqrt` | ✓ | ✓ | ✓ | ✓ | ⚠ |
| `squeeze` | ✓ | ✓ | ✓ | ✓ | - |
| `sub` | ✓ | ✓ | ✓ | ✓ | ⚠ |
| `tan` | ✓ | ✓ | ✓ | ✓ | - |
| `tanh` | ✓ | ✓ | ✓ | ✓ | ⚠ |
| `tile` | ✓ | ✓ | ✓ | ✓ | - |
| `transpose` | ✓ | ✓ | ✓ | ✓ | ⚠ |
| `triangular` | ✓ | ✓ | ✓ | ✓ | - |
| `unsqueeze` | ✓ | ✓ | ✓ | ✓ | - |
| `where` | ✓ | ✓ | ✓ | ✓ | - |

**WPT Data Status:**
- ✓ = Test data populated (17 test cases)
- ⚠ = Test data file exists but empty
- `-` = No WPT test file found

### Deferred Operations

**Rationale:** Each RNN operation requires 10-15 parameters with complex shape inference (~2000-3000 LOC total). Active [W3C discussion](https://github.com/webmachinelearning/webnn/issues/453) about removing these in favor of lower-level primitives. Modern ML trends favor Transformer architectures over LSTM/GRU.

---

## Summary Statistics

```
WebNN Specification Coverage:
  Total Operations in Spec:      ~95
  Fully Implemented:              85 (89%)
  Deferred (RNN):                  4 (lstm, lstmCell, gru, gruCell)
  Remaining:                      ~6 (specialized activations)

Implementation Status:
  Shape Inference:                85/85 ✓ (100%)
  Python API:                     85/85 ✓ (100%)
  ONNX Backend:                   85/85 ✓ (100%)
  CoreML MLProgram:               85/85 ✓ (100%)

Test Coverage:
  Python API Tests:               260 tests (skipped - runtime deps)
  WPT Test Infrastructure:        ✓ Complete (converter + runner)
  WPT Conformance Files:          54 operations
  WPT Validation Files:           17 operations
  WPT Data Populated:             1/71 ✓ (relu with 17 test cases)
  WPT Data Pending:               70/71 (files exist but empty)
```

---

## WPT Integration Status

### What Exists

✓ **Infrastructure:**
- `tests/wpt_data/` directory with conformance/ and validation/ subdirectories
- `tests/test_wpt_conformance.py` - Test runner framework
- `tests/wpt_utils.py` - ULP distance calculation, tolerance checking
- `scripts/convert_wpt_tests.py` - Python converter
- `scripts/extract_wpt_tests.js` - Node.js extraction script (NEW)
- `scripts/update_wpt_tests.sh` - Update automation script

✓ **Test Data Files:**
- 54 conformance test JSON files created
- 17 validation test JSON files created
- Files include metadata: operation name, WPT version, commit SHA, source file

✓ **Test Data Converter:**
- Node.js-based JavaScript parser working
- Successfully extracts test arrays from WPT files
- Validated with relu operation (17 test cases)

⚠ **Current Gap:**
- 1/54 conformance files populated (relu)
- 0/17 validation files populated
- Remaining files have empty "tests": [] arrays
- Need to download/clone full WPT repository for bulk conversion

### Test Status

**Before Converter Fix:**
- pytest shows: `54 skipped` with "no_tests" reason
- All test data files had empty "tests": [] arrays

**After Converter Fix (2025-12-13):**
- pytest shows: `18 collected` for relu (17 test cases + 1 leaky_relu still empty)
- relu.json now has 17 valid test cases covering float32, float16, int8, int32, int64
- Tests properly parameterized but skipped due to missing ONNX Runtime (expected)

---

## Next Steps (Prioritized)

### Priority 1: Complete WPT Test Data Conversion (IN PROGRESS)

**Goal:** Populate remaining WPT test data files with actual test cases from upstream WPT repository

**Status:** ✓ Converter working, 1/54 files converted

**Remaining Tasks:**

1. **Clone WPT repository**
   ```bash
   git clone https://github.com/web-platform-tests/wpt.git ~/wpt
   ```

2. **Convert Tier 1 operations** (28 remaining)
   ```bash
   python scripts/convert_wpt_tests.py \
     --wpt-repo ~/wpt \
     --operations add,sub,mul,div,matmul,pow,sigmoid,tanh,softmax,reduce_sum,reduce_mean \
     --output tests/wpt_data
   ```

   Priority operations:
   - Binary: add, sub, mul, div, matmul, pow (6)
   - Activations: sigmoid, tanh, softmax (3)
   - Reductions: reduce_sum, reduce_mean, reduce_max, reduce_min, reduce_product, reduce_l1, reduce_l2, reduce_log_sum, reduce_log_sum_exp, reduce_sum_square (10)
   - Pooling: average_pool2d, max_pool2d (2)
   - Convolution: conv2d, conv_transpose2d (2)
   - Normalization: batch_normalization, instance_normalization, layer_normalization (3)
   - Shape: reshape (1)

3. **Verify converted test data**
   ```bash
   pytest tests/test_wpt_conformance.py --collect-only
   ```
   - Should show 100+ test cases collected

**Expected Outcome:**
- 29/54 conformance files populated with test data
- 100-200 test cases ready for execution
- Tests skipped only due to runtime dependencies (ONNX Runtime, CoreML)

**Estimated Effort:** 2-3 hours (mostly download/conversion time)

---

### Priority 2: Enable Python API Tests (MEDIUM IMPACT)

**Goal:** Diagnose why 260 Python API tests are skipped and enable execution

**Current Issue:** All Python API tests skipped, likely due to missing ONNX Runtime or other dependencies.

**Action Items:**
1. **Investigate skip conditions**
   ```bash
   pytest tests/test_python_api.py -v --collect-only
   ```
   - Identify why tests are marked as skipped
   - Check for missing pytest markers (e.g., `pytest.mark.asyncio` warning)

2. **Fix runtime dependencies**
   - Ensure ONNX Runtime properly installed: `pip install onnxruntime`
   - Verify `webnn` Python module built: `maturin develop --features python`
   - Check for feature flags or environment variables required

3. **Run tests and document results**
   ```bash
   pytest tests/test_python_api.py -v
   cargo test --lib
   ```

**Expected Outcome:**
- Python API tests passing (or failing with actionable errors)
- Clear documentation of which tests require specific backends
- Skipped tests only for unavailable backends (TensorRT on macOS, CoreML on Linux)

**Estimated Effort:** 4-6 hours

---

### Priority 3: Document Remaining Operations (LOW IMPACT)

**Goal:** Complete WebNN specification coverage analysis

**Action Items:**
1. **Identify remaining ~6 operations** from WebNN spec not yet implemented
2. **Assess priority** based on:
   - Usage in popular models (BERT, ResNet, etc.)
   - Complexity of implementation
   - Backend support availability
3. **Update TODO.txt** with findings

**Expected Outcome:**
- Clear roadmap for reaching 95/95 (100%) operation coverage
- Priority ranking for next implementation phase

**Estimated Effort:** 2-3 hours

---

### Priority 4: CI/CD Integration (MEDIUM IMPACT)

**Goal:** Automate WPT tests in continuous integration pipeline

**Prerequisites:** Priority 1 must be complete (WPT test data populated)

**Action Items:**
1. **Add WPT tests to CI workflow** (`.github/workflows/`)
   - Run on every PR
   - Generate coverage report
   - Fail build on test failures
2. **Create test matrix**
   - Test on multiple platforms (Linux, macOS, Windows)
   - Test with different backends (ONNX CPU, ONNX GPU, CoreML)
3. **Add status badges** to README.md

**Expected Outcome:**
- Automated validation of every code change
- Visible test status for contributors
- Regression prevention

**Estimated Effort:** 4-6 hours (after Priority 1 complete)

---

## Testing Strategy Details

### WPT Test Structure

**Conformance Tests** (`tests/wpt_data/conformance/`)
- Validate numerical correctness of operations
- Use ULP (Units in Last Place) or ATOL (absolute tolerance) based checking
- Test multiple input shapes, data types, and parameter combinations

**Validation Tests** (`tests/wpt_data/validation/`)
- Validate parameter constraints and error handling
- Test invalid inputs produce correct error messages
- Test boundary conditions

### Tolerance Checking

The `wpt_utils.py` module implements WPT-compatible precision validation:

```python
def ulp_distance(a: float, b: float, dtype: str) -> int:
    """Calculate ULP distance between two floating-point values"""
    # Handles float32 and float16
    # Returns number of representable values between a and b
```

**Per-Operation Tolerances:**
- `relu`: 0 ULP (exact)
- `sigmoid`: 34 ULP (float32), 3 ULP (float16)
- `tanh`: 44 ULP (float32), 4 ULP (float16)
- `reduce_*`: Varies based on input size (accumulation error)

### Running Tests

```bash
# Run all WPT conformance tests (when data populated)
pytest tests/test_wpt_conformance.py -v

# Run tests for specific operation
pytest tests/test_wpt_conformance.py -k "reduce_sum" -v

# Run with coverage report
pytest tests/test_wpt_conformance.py --cov=webnn --cov-report=html

# Run Python API tests (when runtime available)
pytest tests/test_python_api.py -v

# Run all tests
make python-test
```

---

## References

- **W3C WebNN Specification:** https://www.w3.org/TR/webnn/
- **WPT WebNN Tests:** https://github.com/web-platform-tests/wpt/tree/master/webnn
- **Local WebNN Spec Reference:** `docs/webnn-spec-reference.md`
- **API Reference:** `docs/api-reference.md`
- **Development Guide:** `docs/development.md`

---

## Revision History

- **2025-12-13:**
  - Reorganized into single alphabetically sorted table with simple check icons (✓)
  - Fixed WPT test data converter with Node.js-based extraction
  - Successfully converted relu operation (17 test cases)
  - Updated status: converter working, 1/54 files populated
- **2025-12-13:** Merged operator-status.md and wpt-integration-plan.md; identified WPT test data gap as critical blocker
- **2025-12-08:** 85 operations fully implemented; CoreML end-to-end execution verified
- **2025-12-07:** WPT test infrastructure created; test data files initialized (empty)

---

**Document Status:** Living Document - Update after major implementation milestones
