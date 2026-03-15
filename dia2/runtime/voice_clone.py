from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, TYPE_CHECKING, Union

import numpy as np
import torch

from ..generation import PrefixConfig
from .audio_io import encode_audio_tokens, load_mono_audio
from .state_machine import Entry

if TYPE_CHECKING:  # pragma: no cover
    from .context import RuntimeContext


@dataclass
class WhisperWord:
    text: str
    start: float
    end: float


@dataclass
class PrefixPlan:
    entries: List[Entry]
    new_word_steps: List[int]
    aligned_tokens: torch.Tensor
    aligned_frames: int
    transcripts: Dict[str, Any]


def parse_transcript(
    transcript: Union[str, List[Any], Dict[str, Any]],
) -> List[WhisperWord]:
    """Parse a user-provided transcript into a list of :class:`WhisperWord`.

    *transcript* may be either:

    * A ``str`` path to a JSON file, or
    * A ``list`` of dicts already loaded in memory, or
    * A ``dict`` with an ``"audio_hash"`` and ``"words"`` key (as returned by
      a previous generation call).

    Each word entry must have ``"text"``, ``"start"`` (seconds), and ``"end"``
    (seconds) keys.
    """
    if isinstance(transcript, str):
        with open(transcript, "r") as f:
            data = json.load(f)
    else:
        data = transcript

    # Accept the dict format returned by generate()/generate_stream()
    if isinstance(data, dict):
        data = data.get("words", [])

    if not isinstance(data, list):
        raise ValueError("Transcript must be a JSON array of word objects")

    words: List[WhisperWord] = []
    for entry in data:
        text = str(entry.get("text", "")).strip()
        if not text:
            continue
        words.append(
            WhisperWord(
                text=text,
                start=float(entry["start"]),
                end=float(entry["end"]),
            )
        )
    return words


def _hash_audio_file(audio_path: str) -> str:
    """Compute a SHA-256 hash of an audio file's contents."""
    h = hashlib.sha256()
    with open(audio_path, "rb") as f:
        for block in iter(lambda: f.read(65536), b""):
            h.update(block)
    return f"sha256:{h.hexdigest()}"


def _build_transcript_dict(
    audio_path: str, words: List[WhisperWord]
) -> Dict[str, Any]:
    """Build a transcript dict with audio hash and word list."""
    return {
        "audio_hash": _hash_audio_file(audio_path),
        "words": [
            {"text": w.text, "start": w.start, "end": w.end} for w in words
        ],
    }


def build_prefix_plan(
    runtime: "RuntimeContext",
    prefix: Optional[PrefixConfig],
    *,
    transcribe_fn: Optional[Callable[[str, torch.device], List[WhisperWord]]] = None,
    load_audio_fn: Optional[Callable[[str, int], np.ndarray]] = None,
    encode_fn: Optional[Callable[[np.ndarray], torch.Tensor]] = None,
) -> Optional[PrefixPlan]:
    if prefix is None:
        return None
    if not prefix.speaker_1:
        if prefix.speaker_2:
            raise ValueError("speaker_2 requires speaker_1 to be provided")
        return None

    default_transcribe = transcribe_fn or (lambda path, device: transcribe_words(path, device))
    load_audio = load_audio_fn or (lambda path, sr: load_mono_audio(path, sr))
    encode_audio = encode_fn or (lambda audio: encode_audio_tokens(runtime.mimi, audio))

    # If a pre-computed transcript is provided for a speaker, build a
    # transcribe function that returns the parsed words directly —
    # skipping the Whisper model entirely.
    if prefix.transcript_speaker_1 is not None:
        spk1_words = parse_transcript(prefix.transcript_speaker_1)

        def transcribe_spk1(_p: str, _d: torch.device) -> List[WhisperWord]:
            return spk1_words
    else:
        transcribe_spk1 = default_transcribe

    entries1, steps1, tokens1, words1 = _process_prefix_audio(
        runtime=runtime,
        audio_path=prefix.speaker_1,
        speaker_token=runtime.constants.spk1,
        transcribe=transcribe_spk1,
        load_audio=load_audio,
        encode_audio=encode_audio,
    )
    offset = 3  # Match legacy BOS/PAD offset
    entries = list(entries1)
    new_word_steps = [step + offset for step in steps1]
    audio_tokens = tokens1.to(runtime.device)

    transcripts: Dict[str, Any] = {
        "speaker_1": _build_transcript_dict(prefix.speaker_1, words1),
    }

    if prefix.speaker_2:
        if prefix.transcript_speaker_2 is not None:
            spk2_words = parse_transcript(prefix.transcript_speaker_2)

            def transcribe_spk2(_p: str, _d: torch.device) -> List[WhisperWord]:
                return spk2_words
        else:
            transcribe_spk2 = default_transcribe

        entries2, steps2, tokens2, words2 = _process_prefix_audio(
            runtime=runtime,
            audio_path=prefix.speaker_2,
            speaker_token=runtime.constants.spk2,
            transcribe=transcribe_spk2,
            load_audio=load_audio,
            encode_audio=encode_audio,
        )
        spk1_frames = audio_tokens.shape[-1]
        new_word_steps.extend(step + spk1_frames for step in steps2)
        entries.extend(entries2)
        audio_tokens = torch.cat([audio_tokens, tokens2.to(runtime.device)], dim=1)
        transcripts["speaker_2"] = _build_transcript_dict(prefix.speaker_2, words2)

    return PrefixPlan(
        entries=entries,
        new_word_steps=new_word_steps,
        aligned_tokens=audio_tokens,
        aligned_frames=audio_tokens.shape[-1],
        transcripts=transcripts,
    )


def _process_prefix_audio(
    runtime: "RuntimeContext",
    audio_path: str,
    speaker_token: int,
    *,
    transcribe: Callable[[str, torch.device], List[WhisperWord]],
    load_audio: Callable[[str, int], np.ndarray],
    encode_audio: Callable[[np.ndarray], torch.Tensor],
) -> tuple[List[Entry], List[int], torch.Tensor, List[WhisperWord]]:
    words = transcribe(audio_path, runtime.device)
    entries, steps = words_to_entries(
        words=words,
        tokenizer=runtime.tokenizer,
        speaker_token=speaker_token,
        frame_rate=runtime.frame_rate,
    )
    audio = load_audio(audio_path, runtime.mimi.sample_rate)
    tokens = encode_audio(audio)
    return entries, steps, tokens, words


def transcribe_words(
    audio_path: str,
    device: torch.device,
    language: Optional[str] = None,
) -> List[WhisperWord]:
    import whisper_timestamped as wts  # Imported lazily

    model = wts.load_model("openai/whisper-large-v3", device=str(device))
    result = wts.transcribe(model, audio_path, language=language)

    words: List[WhisperWord] = []
    for segment in result.get("segments", []):
        for word in segment.get("words", []):
            text = (word.get("text") or word.get("word") or "").strip()
            if not text:
                continue
            words.append(
                WhisperWord(
                    text=text,
                    start=float(word.get("start", 0.0)),
                    end=float(word.get("end", 0.0)),
                )
            )
    return words


def words_to_entries(
    *,
    words: Sequence[WhisperWord],
    tokenizer,
    speaker_token: int,
    frame_rate: float,
) -> tuple[List[Entry], List[int]]:
    entries: List[Entry] = []
    new_word_steps: List[int] = []
    if not words:
        return entries, new_word_steps

    convert = getattr(tokenizer, "convert_tokens_to_ids", None)
    speaker_prefix: Optional[str] = None
    if callable(convert):
        s1_id = convert("[S1]")
        s2_id = convert("[S2]")
        if speaker_token == s1_id:
            speaker_prefix = "[S1]"
        elif speaker_token == s2_id:
            speaker_prefix = "[S2]"
    pending_prefix: Optional[str] = speaker_prefix
    current_pos = 0

    for idx, word in enumerate(words):
        tokens = _encode_word(word.text, tokenizer, pending_prefix)
        pending_prefix = None
        start_frame = max(current_pos + 1, int(round(word.start * frame_rate)))
        end_frame = start_frame + len(tokens)
        new_word_steps.append(start_frame - 1)

        if idx < len(words) - 1:
            next_start = int(round(words[idx + 1].start * frame_rate))
            next_word_start = max(end_frame + 1, next_start)
        else:
            end_time = int(round(words[-1].end * frame_rate))
            next_word_start = max(end_frame + 1, end_time)

        padding = max(0, next_word_start - start_frame - 1)
        entries.append(Entry(tokens=tokens, text=word.text, padding=padding))
        current_pos = end_frame

    return entries, new_word_steps


def _encode_word(text: str, tokenizer, prefix: Optional[str]) -> List[int]:
    if prefix:
        return tokenizer.encode(f"{prefix} {text}", add_special_tokens=False)
    return tokenizer.encode(text, add_special_tokens=False)


__all__ = [
    "PrefixPlan",
    "WhisperWord",
    "build_prefix_plan",
    "parse_transcript",
    "transcribe_words",
    "words_to_entries",
]
