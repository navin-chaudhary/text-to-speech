from __future__ import annotations

import os
import tempfile
from contextlib import contextmanager

import speech_recognition as sr


def merge_with_overlap(existing: str, incoming: str) -> str:
    existing = existing.strip()
    incoming = incoming.strip()
    if not incoming:
        return existing
    if not existing:
        return incoming

    existing_words = existing.split()
    incoming_words = incoming.split()
    max_overlap = min(len(existing_words), len(incoming_words), 8)

    overlap = 0
    for size in range(max_overlap, 0, -1):
        if existing_words[-size:] == incoming_words[:size]:
            overlap = size
            break

    if overlap == len(incoming_words):
        return existing

    merged_words = existing_words + incoming_words[overlap:]
    return " ".join(merged_words)


@contextmanager
def _temp_wav_path(audio_bytes: bytes):
    path = ""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(audio_bytes)
            path = tmp.name
        yield path
    finally:
        if path and os.path.exists(path):
            os.remove(path)


def transcribe_audio_bytes(recognizer: sr.Recognizer, audio_bytes: bytes) -> str:
    with _temp_wav_path(audio_bytes) as temp_path:
        with sr.AudioFile(temp_path) as source:
            audio = recognizer.record(source)
        return recognizer.recognize_google(audio).strip()
