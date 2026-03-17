<div align="center">

# Dia 2.1

**A streaming dialogue TTS engine — forked by [Generative Experiences Company](https://github.com/khariha/dia2)**

<a href="https://huggingface.co/nari-labs/Dia2-2B"><img src="https://img.shields.io/badge/HF%20Repo-Dia2--2B-orange?style=for-the-badge"></a>
<a href="https://github.com/nari-labs/dia2"><img src="https://img.shields.io/badge/Upstream-nari--labs%2Fdia2-333?style=for-the-badge&logo=github"></a>
<a href="https://github.com/nari-labs/dia2/blob/main/LICENSE"><img src="https://img.shields.io/badge/License-Apache_2.0-blue.svg?style=for-the-badge"></a>

</div>

Dia 2.1 builds on [Nari Labs' Dia2](https://github.com/nari-labs/dia2) with features designed for real-time, production-grade dialogue generation. All upstream performance improvements are merged.

## What's New in 2.1

### Streaming Audio Generation
`generate_stream()` yields PCM audio chunks as the model generates, enabling real-time playback before generation completes.

```python
stream = dia.generate_stream(
    "[S1] Hello!\n[S2] Hey, how's it going?",
    prefix_speaker_1="speaker1.wav",
    prefix_speaker_2="speaker2.wav",
    chunk_frames=1,
)
for chunk in stream:
    play(chunk.cpu().numpy())  # ~80 ms per chunk at 12.5 fps
```

Each chunk is a `torch.Tensor` with shape `[N_samples]` (float32, range `[-1, 1]`). The `chunk_frames` parameter controls how many Mimi frames (~80 ms each) are batched per yield — larger values reduce overhead at the cost of higher initial latency.

### Stateless Transcript Caching

When using voice conditioning (prefix speakers), Dia2 runs Whisper to transcribe each prefix audio file. This is slow. Dia 2.1 returns the Whisper transcript and a SHA-256 audio hash with every generation response, so you can pass them back on subsequent calls to **skip Whisper entirely**.

**First call — Whisper runs, transcript returned:**
```python
stream = dia.generate_stream(script, prefix_speaker_1="speaker.wav")
for chunk in stream:
    play(chunk.cpu().numpy())

# Grab the transcript (available immediately, before or after iteration)
transcript_1 = stream.prefix_transcripts["speaker_1"]
```

**Second call — Whisper skipped:**
```python
stream = dia.generate_stream(
    script,
    prefix_speaker_1="speaker.wav",
    prefix_speaker_1_transcript=transcript_1,
)
for chunk in stream:
    play(chunk.cpu().numpy())
```

This is fully **stateless** — no files are cached on disk. The caller manages their own transcript data. The same pattern works with `generate()` via `result.prefix_transcripts`.

### Transcript & Hash Return Format

Both `generate()` and `generate_stream()` return transcript data in this format:

```python
{
    "speaker_1": {
        "audio_hash": "sha256:abc123...",
        "words": [
            {"text": "Hello", "start": 0.0, "end": 0.35},
            {"text": "world", "start": 0.4, "end": 0.72},
        ]
    },
    "speaker_2": {  # only present if prefix_speaker_2 was provided
        "audio_hash": "sha256:def456...",
        "words": [...]
    }
}
```

- `audio_hash` — SHA-256 of the prefix audio file, for verifying the transcript matches
- `words` — Whisper word-level timestamps (same format Whisper produces)

For `generate()`, access via `result.prefix_transcripts`. For `generate_stream()`, access via `stream.prefix_transcripts`.

### AudioStream Wrapper

`generate_stream()` returns an `AudioStream` object instead of a bare generator. This allows metadata (like `prefix_transcripts`) to be available before iteration begins:

```python
stream = dia.generate_stream(script, prefix_speaker_1="speaker.wav")

# Available immediately — no need to iterate first
print(stream.prefix_transcripts)

# Then iterate as normal
for chunk in stream:
    process(chunk)
```

### Merged Upstream Performance Improvements

All performance optimizations from the upstream Dia2 repo are included:

- **RotaryEmbedding sin/cos caching** — pre-computed at init, table lookup instead of per-step computation
- **CPU-GPU sync elimination** — replaced `if tensor.any().item()` with `torch.where()` to avoid GPU stalls
- **On-device model initialization** — layers created directly on the target device, avoiding CPU-to-GPU transfer
- **Lazy safetensors loading** — weights loaded tensor-by-tensor for lower peak memory
- **`torch.compile` support** — optional `use_torch_compile=True` for additional speed on supported hardware

## Quickstart

> **Requirement** — install [uv](https://docs.astral.sh/uv/) and use CUDA 12.8+
> drivers. All commands below run through `uv run …` as a rule.

1. **Install dependencies (one-time):**
   ```bash
   uv sync
   ```
2. **Prepare a script:** edit `input.txt` using `[S1]` / `[S2]` speaker tags.
3. **Generate audio:**
   ```bash
   uv run -m dia2.cli \
     --hf nari-labs/Dia2-2B \
     --input input.txt \
     --cfg 6.0 --temperature 0.8 \
     --cuda-graph --verbose \
     output.wav
   ```
   The first run downloads weights/tokenizer/Mimi. The CLI auto-selects CUDA when available (otherwise CPU) and defaults to bfloat16 precision — override with `--device` / `--dtype` if needed.
4. **Conditional Generation (recommended for stable output):**
   ```bash
   uv run -m dia2.cli \
     --hf nari-labs/Dia2-2B \
     --input input.txt \
     --prefix-speaker-1 example_prefix1.wav \
     --prefix-speaker-2 example_prefix2.wav \
     --cuda-graph --verbose \
     output_conditioned.wav
   ```
5. **Gradio UI:**
   ```bash
   uv run gradio_app.py
   ```

### Programmatic Usage

```python
from dia2 import Dia2, GenerationConfig, SamplingConfig

dia = Dia2.from_repo("nari-labs/Dia2-2B", device="cuda", dtype="bfloat16")
config = GenerationConfig(
    cfg_scale=2.0,
    audio=SamplingConfig(temperature=0.8, top_k=50),
    use_cuda_graph=True,
)
result = dia.generate("[S1] Hello Dia2!", config=config, output_wav="hello.wav", verbose=True)
```

## Hugging Face

| Variant | Repo |
| --- | --- |
| Dia2-1B | [`nari-labs/Dia2-1B`](https://huggingface.co/nari-labs/Dia2-1B) |
| Dia2-2B | [`nari-labs/Dia2-2B`](https://huggingface.co/nari-labs/Dia2-2B) |

## License & Attribution

Licensed under [Apache 2.0](LICENSE). Forked from [nari-labs/dia2](https://github.com/nari-labs/dia2). All third-party assets (Kyutai Mimi codec, etc.) retain their original licenses.

## Disclaimer

This project offers a high-fidelity speech generation model intended for research and educational use. The following uses are **strictly forbidden**:

- **Identity Misuse**: Do not produce audio resembling real individuals without permission.
- **Deceptive Content**: Do not use this model to generate misleading content (e.g. fake news)
- **Illegal or Malicious Use**: Do not use this model for activities that are illegal or intended to cause harm.

By using this model, you agree to uphold relevant legal standards and ethical responsibilities. We **are not responsible** for any misuse and firmly oppose any unethical usage of this technology.

## Acknowledgements
- Original model and research by [Nari Labs](https://github.com/nari-labs)
- [TPU Research Cloud](https://sites.research.google/trc/about/) for training compute
- Inspired by [KyutaiTTS](https://kyutai.org/next/tts) and [Sesame](https://www.sesame.com/research/crossing_the_uncanny_valley_of_voice)
