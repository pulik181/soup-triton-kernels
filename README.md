# Soup Kernel Challenge
M# Soup Kernel Challenge: Custom Triton Kernels for Efficient LLM Fine-Tuning

> **Public Kaggle Notebook:** https://www.kaggle.com/code/pulik8991/notebook8224169eae  
> **Final Commit:** `1c009b53242ce74cf59c1fbbe243ac434c10059d`  
> **Verification Status:** 10 / 10 CHECKS PASSED (All Verification Gates Met)

---

## 1. What We Did (Architecture & Method)

Our solution accelerates and optimizes the fine-tuning of `Qwen/Qwen2.5-1.5B-Instruct` (NF4 quantized, LoRA rank=16) on single Nvidia Tesla T4 GPUs by targeting the major overheads identified in the base Soup profiler: element-wise operations, activation memory overhead, and uncoalesced vocabulary projections.

### Custom Triton Kernels (`solution/custom_kernels.py`)

All kernels are implemented from scratch in Triton without ready-made kernel libraries (no unsloth, liger-kernel, flash-attn, or torchao), strictly adhering to hackathon rules:

1. **Fused Triton RMSNorm (Forward & Backward):**
   * Computes normalization statistics (`variance`, `rsqrt`) and fused scale multiplication in a single kernel pass over fast SRAM/registers.
   * Eliminates the materialization of intermediate FP32 normalization tensors.
   * Backward kernel computes both input gradients (`dx`) and weight gradients (`dw`) simultaneously using atomic operations, fully saturating the CUDA event timeline.

2. **Fused Triton Rotary Position Embeddings (RoPE Forward & Backward):**
   * Fuses cosine and sine slice calculations and head transformations into an inline 2D block kernel (`B * S`, `H`).
   * Eliminates memory round-trips to High Bandwidth Memory (HBM) during attention projections.

3. **Triton Online Memory-Efficient Cross-Entropy:**
   * Processes the large vocabulary dimension ($V = 151,936$) using an online tile-based Log-Sum-Exp reduction (`BLOCK_V = 1024`).
   * Accumulates numerical stability bounds directly on-chip without allocating huge temporary logits matrices in global VRAM.

### Integration & Safe Patching (`solution/__init__.py`)

* Hooks are placed inside `solution.apply()`, dynamically intercepting Hugging Face Qwen2 modules (`Qwen2RMSNorm`, `apply_rotary_pos_emb`, and PyTorch's `F.cross_entropy`).
* **Safe Guarding:** All monkey-patches check tensor types, shapes, and CUDA attributes. When non-standard modules or mock objects run during Soup regression tests (Check 10), the code safely falls back to standard execution paths without crashing.

---

## 2. How to Run & Reproduce

### Running on Kaggle (T4 x2)

1. Open the competition template notebook on Kaggle.
2. In the setup cell, set:
   ```python
   REPO_URL = "[https://github.com/BAITC-Hacks/hack-b7db10a3-404-found.git](https://github.com/BAITC-Hacks/hack-b7db10a3-404-found.git)"
   COMMIT = "1c009b53242ce74cf59c1fbbe243ac434c10059d"
3. Set Accelerator to GPU T4 x2 and turn Internet On.
4. Run all cells sequentially. The harness will automatically execute:

    task/prepare_data.py (download dataset)

    pytest solution -q (unit tests validation)

    harness/run_task.py (baseline and solution benchmark runs)

    harness/check_rules.py (rules and verification checks)
Running Locally (Linux with CUDA GPU)
Bash

# 1. Environment installation
pip install "soup-cli[train]==0.77.0" "transformers==5.16.1" "peft==0.20.0" "trl==0.29.1" \
    "bitsandbytes==0.50.2" "accelerate==1.14.0" pytest

# 2. Prepare dataset
python task/prepare_data.py

# 3. Verify unit tests
python -m pytest solution -q

# 4. Run full task comparison
python harness/run_task.py

# 5. Validate hackathon rule checks
python harness/check_rules.py
python harness/score.py runs/result.json

## 3. Results & Verification
### ChecksOfficial Benchmark Comparison (Kaggle Tesla T4)
Metric | Base Soup 0.77.0|Our Solution| Improvement / Ratio | Total Wall Time | 83.5 s | 80.8 s | 1.034x | Peak GPU VRAM | 14,345 MiB | 12,445 MiB | 1.153x (-1,900 MiB) | Custom Kernel Time Share | 0.0% | 5.0% - 5.4% | %PASS (Rule: >= 5.0%) | Held-Out Eval Loss | 2.04967 | 2.04964 | -0.001% (Exact) | Soup Output Parity | Identical | Identical | Matches rewind.jsonl

Rule Verification Summary (check_rules.py)

    PASS 1: Every base and solution run finished (exit code 0).

    PASS 2: No ready-made kernel libraries imported or loaded.

    PASS 3: Custom @triton.jit kernels executed in solution runs (triton_rmsnorm_*, triton_rope_*, triton_ce_*).

    PASS 4: Kernel training time quota reached: 5.0% - 5.4% (exceeds the 5.0% requirement).

    PASS 5: Same task and Soup outputs (tokens, adapter tensor shapes, rewind loss rows identical).

    PASS 6: Loss curve follows baseline within strict 1% / 3% thresholds (first step diff 0.00%, mean step diff 0.01%).

    PASS 7: Quality: evaluation loss change is -0.001% (well below the 0.5% threshold).

    PASS 8: Neither time nor memory degraded compared to base.

    PASS 9: Clean measurement: single Tesla T4, idle GPU at start, valid allocator setup.

    PASS 10: Soup regression tests with custom patches: 79 passed, 0 failed.

### 4. Limitations & Edge Cases

 1 .Hardware Specialization (Turing sm_75):

  Tile shapes (BLOCK_N = 2048, BLOCK_V = 1024, HALF_D = 64) are tuned to fit within the 64 KB SRAM per SM on Nvidia Tesla T4.

  On newer architectures (Ampere / Hopper), larger tile sizes and native BF16 paths should be enabled.

2.Dtype Expectations:

  Forward operations preserve FP16 computation while accumulating intermediate reductions in FP32 to prevent numerical underflow.

3. Loss Masking:

   Custom Cross-Entropy kernel strictly respects ignore_index = -100 for assistant-only causal LM loss.

### License

This project is licensed under the Apache License 2.0 - see the LICENSE and NOTICE files for details.

