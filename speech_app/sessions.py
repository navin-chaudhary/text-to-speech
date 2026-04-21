from __future__ import annotations

import asyncio
from uuid import uuid4


class SessionManager:
    """In-memory queues and rolling transcripts for chunked HTTP streaming."""

    def __init__(self) -> None:
        self._queues: dict[str, asyncio.Queue[str]] = {}
        self._transcripts: dict[str, str] = {}

    def create_session(self) -> str:
        session_id = str(uuid4())
        self._queues[session_id] = asyncio.Queue()
        self._transcripts[session_id] = ""
        return session_id

    def get_queue(self, session_id: str) -> asyncio.Queue[str] | None:
        return self._queues.get(session_id)

    def get_transcript(self, session_id: str) -> str:
        return self._transcripts.get(session_id, "")

    def set_transcript(self, session_id: str, text: str) -> None:
        self._transcripts[session_id] = text

    def end_session(self, session_id: str) -> None:
        self._queues.pop(session_id, None)
        self._transcripts.pop(session_id, None)
