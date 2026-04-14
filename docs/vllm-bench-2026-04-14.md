# vllm vs llama.cpp benchmark — 2026-04-14

## Setup
- LFM2.5-1.2B-Instruct base (HF safetensors, not quantized)
- vllm 0.19.0, max_model_len=16384, lora enabled
- llama.cpp llama-server, Q4_K_M GGUF, n-gpu-layers 999
- 4090 24 GB

## Numbers

| Backend | Mode | Throughput |
|---|---|---|
| llama.cpp | sequential generation | ~670 tok/s |
| vllm | sequential generation | ~330 tok/s |
| vllm | batch×8 concurrent (aggregate) | **~1880 tok/s** |

## Implication
For single-request use, llama.cpp wins. For generate-N rerank (G2 / Day 3),
vllm wins decisively — N=8 candidates per scene fit in 1.2s wall-clock vs
6-8s sequential. The roadmap target of "3k tok/s" is in reach with batch=16
or larger (didn't push the limit).

## Decision
- **vllm** for production writer (generate-N requires concurrent batching).
- **llama.cpp** retained for single-shot debugging + when LoRA hot-swap not needed.
- Both servers can coexist on different ports.
