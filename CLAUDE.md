# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Dia2** is a streaming dialogue text-to-speech (TTS) model by Nari Labs. It generates speech without requiring the full text input, enabling real-time audio synthesis for dialogue scenarios using `[S1]`/`[S2]` speaker tags.

## Commands

All commands must run via `uv run` from the `src/` directory.

**Install dependencies (one-time):**
```bash
cd src && uv sync
```

**Install with dev tools:**
```bash
cd src && uv sync --extra dev
```

**Generate audio via CLI:**
```bash
uv run -m dia2.cli --hf nari-labs/Dia2-2B --input input.txt --cfg 6.0 --temperature 0.8 --cuda-graph --verbose output.wav
```

**Generate with voice conditioning:**
```bash
uv run -m dia2.cli --hf nari-labs/Dia2-2B --input input.txt --prefix-speaker-1 example_prefix1.wav --prefix-speaker-2 example_prefix2.wav --cuda-graph --verbose output.wav
```

**Lint:**
```bash
cd src && uv run ruff check .
```

**Type check:**
```bash
cd src && uv run pyright
```

**Launch Gradio web UI:**
```bash
cd src && uv run gradio_app.py
```

## Streaming API

`Dia2.generate_stream()` yields `torch.Tensor` PCM chunks (shape `[N_samples]`, float32 in `[-1, 1]`) as the model generates, enabling real-time playback before generation completes:

```python
for chunk in dia.generate_stream("[S1] Hello Dia2!", chunk_frames=1):
    play(chunk.detach().cpu().numpy())  # ~80 ms per chunk at 12.5 fps
```

**How it works:** Each generation step fills one column of the audio token buffer. Due to `audio_delays` (per-codebook temporal offsets, max = `max_delay` steps), the aligned Mimi frame at index `f` is complete once step `f + max_delay` finishes. After that warmup, one new frame (~1920 samples at 24 kHz) is decodable per step. `stream_generation_loop()` in `generator.py` handles this alignment and yields batches of `chunk_frames` frames decoded by Mimi.

**crop handling:** Frames before `first_word_frame` (initial silence/prefix audio) are buffered in `pre_buffer` until `first_word_frame` is determined (within the first few steps), then discarded — matching `generate()` behavior. The final post-loop frame (`last_step + 1 - max_delay`) is emitted after the loop ends.

## Architecture

The source code lives under `src/dia2/`. The package is organized into three layers:

### Core (`dia2/core/`)
Neural network architecture:
- `model.py` — `Dia2Model`: top-level PyTorch model combining transformer and depformer
- `transformer.py` — Autoregressive transformer decoder for text-conditioned generation
- `depformer.py` — Depth-wise transformer that refines audio tokens across codebook levels
- `layers.py` — Custom attention and MLP layers
- `cache.py` — KV cache for efficient incremental inference
- `precision.py` — Dtype/device management utilities

### Runtime (`dia2/runtime/`)
Generation pipeline components:
- `context.py` — `RuntimeContext` and `build_runtime()`: assembles all components for a generation run
- `generator.py` — Core autoregressive generation loop
- `state_machine.py` — Token streaming state machine that manages text/audio transitions
- `voice_clone.py` — Prefix audio conditioning for speaker control
- `script_parser.py` — Parses dialogue scripts with `[S1]`/`[S2]` tags
- `sampler.py` — Token sampling (top-k, temperature)
- `guidance.py` — Classifier-free guidance (CFG) implementation
- `audio_io.py` — Audio encode/decode helpers

### Audio (`dia2/audio/`)
- `codec.py` — Wrapper around the Kyutai Mimi neural audio codec (~12.5 Hz frame rate)
- `grid.py` — Audio frame alignment utilities

### Public API (`dia2/`)
- `engine.py` — `Dia2` class: main user-facing inference engine (`from_repo`, `generate`)
- `config.py` — Configuration dataclasses loaded from YAML
- `generation.py` — `GenerationConfig`, `SamplingConfig`, `GenerationResult` dataclasses
- `cli.py` — CLI argument parsing; wraps `engine.py`
- `assets.py` — Resolves model weights and configs from HF Hub or local paths
- `gradio_app.py` — Gradio web UI (at `src/gradio_app.py`)

### Data Flow

```
input.txt → script_parser → transformer (text tokens)
                                  ↓
                          depformer (audio tokens × codebook levels)
                                  ↓
                          Mimi codec decode → waveform
```

CFG runs the transformer twice (conditioned + unconditioned) and interpolates logits by `cfg_scale`. Voice conditioning prepends encoded prefix audio tokens before generation begins.

## Key Constraints

- **Maximum generation length:** 1500 context steps (~2 minutes of audio)
- **Language:** English only
- **CUDA 12.8+** required for GPU acceleration; macOS uses CPU/MPS
- **Python 3.10+** required
- PyTorch is sourced from the custom `pytorch-cu128` index on Linux/Windows; macOS gets the default index

## Model Variants

| Variant | HF Repo |
|---------|---------|
| Dia2-1B | `nari-labs/Dia2-1B` |
| Dia2-2B | `nari-labs/Dia2-2B` |

First run downloads weights, tokenizer, and Mimi codec from Hugging Face automatically.
