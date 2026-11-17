# Submission: 404 Found

Copy this file to the root of your repository as SUBMISSION.md and fill it in.

## Links
- Repository, final commit: https://github.com/BAITC-Hacks/hack-b7db10a3-404-found/commit/1c009b53242ce74cf59c1fbbe243ac434c10059d
- Public Kaggle notebook, the saved version (COMMIT in it = the sha above; the same link must be in your README):
  https://www.kaggle.com/code/pulik8991/notebook8224169eae
- Demo video (up to 3 minutes): https://drive.google.com/file/d/1Oys5k6l3zy4XYR39hFKbqp0PQoWhjrIk/view?usp=sharing
- Presentation: https://github.com/BAITC-Hacks/hack-b7db10a3-404-found/blob/main/Soup_Kernel_Challenge_404_Found_v2.pdf

## Team (up to 4 people)
| Name | Role | GitHub | Kaggle | Contact (captain) |
|---|---|---|---|---|
| Aslan Kudiyarov | Lead GPU & ML Systems Engineer (Triton Kernels, Profiling & Harness Validation) | @pulik181 | pulik8991 | @Pulik3214 (captain) |
| Shairdin Tuigunov | Presentation, Benchmark Reproduction & Demo Video Delivery | @dintuig | dintuig | @TuigunovDin |

## Our kernels
| Kernel (@triton.jit name) | File | What it replaces | Share of the training time (check 4) |
|---|---|---|---|
| `triton_rmsnorm_fwd_kernel` | `solution/custom_kernels.py` | Fused Qwen2RMSNorm forward pass | ~1.8% |
| `triton_rmsnorm_bwd_kernel` | `solution/custom_kernels.py` | Fused Qwen2RMSNorm backward + atomic dw | ~1.4% |
| `triton_rope_fwd_kernel` | `solution/custom_kernels.py` | Inline 2D RoPE attention projection | ~0.9% |
| `triton_rope_bwd_kernel` | `solution/custom_kernels.py` | RoPE attention backward pass | ~0.5% |
| `triton_ce_fwd_kernel` | `solution/custom_kernels.py` | Tiled memory-efficient online Cross-Entropy | ~0.4% |

Ideas taken from other kernels, with links (we wrote the code ourselves):
- Online reduction / running max logic inspired by FlashAttention principles (Dao et al.).
- Fused normalization pattern based on standard Triton tutorials.

## Results on Kaggle T4 (from runs/result.json and runs/checks.json of the notebook above)
| | Base Soup | Ours | Ratio |
|---|---|---|---|
| Time of soup train, s | 83.5 s | 80.8 s | speedup 1.034x |
| Peak GPU memory, MiB | 14,345 MiB | 12,445 MiB | base/ours 1.153x |
| Eval loss (public rows) | 2.04967 | 2.04964 | -0.001 % vs base |

- [x] All 10 checks of harness/check_rules.py passed
- [x] README.md rewritten for our solution: what we did, how to run it, results, limits

## Presentation covers
Which operations you sped up and why, how the kernels work, results against the base (speed, memory, quality), limits and next steps.

## Limitations
What does not work yet, where the speedup stops, what you would do next:
- Tile block shapes (BLOCK_N=2048, BLOCK_V=1024) are specialized for Turing sm_75 architecture (Tesla T4 64 KB SRAM limit)[cite: 10].
- Cross-entropy supports causal LM loss calculation with ignore_index=-100 masking[cite: 10].
