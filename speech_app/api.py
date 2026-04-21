from __future__ import annotations

import asyncio
import json
from functools import lru_cache

import speech_recognition as sr
import uvicorn
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from starlette.staticfiles import StaticFiles

from speech_app.paths import STATIC_DIR, TEMPLATES_DIR
from speech_app.sessions import SessionManager
from speech_app.transcription import merge_with_overlap, transcribe_audio_bytes

app = FastAPI(title="Browser Speech-to-Text")
recognizer = sr.Recognizer()
sessions = SessionManager()


@lru_cache(maxsize=1)
def _index_html() -> str:
    return (TEMPLATES_DIR / "index.html").read_text(encoding="utf-8")


app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/", response_class=HTMLResponse)
async def index() -> str:
    return _index_html()


@app.post("/transcribe")
async def transcribe(request: Request) -> JSONResponse:
    data = await request.body()
    if not data:
        return JSONResponse({"error": "No audio data received."}, status_code=400)

    try:
        text = transcribe_audio_bytes(recognizer, data)
        return JSONResponse({"text": text})
    except sr.UnknownValueError:
        return JSONResponse({"error": "Could not understand audio."}, status_code=422)
    except sr.RequestError:
        return JSONResponse({"error": "Google API unavailable."}, status_code=503)
    except Exception as exc:
        return JSONResponse({"error": f"Processing failed: {exc}"}, status_code=500)


@app.websocket("/ws/transcribe")
async def transcribe_websocket(websocket: WebSocket) -> None:
    await websocket.accept()
    running_transcript = ""
    try:
        while True:
            message = await websocket.receive()
            msg_type = message.get("type", "")
            if msg_type == "websocket.disconnect":
                break

            audio_bytes = message.get("bytes")
            text_payload = message.get("text")

            if text_payload:
                try:
                    payload = json.loads(text_payload)
                except json.JSONDecodeError:
                    continue
                if payload.get("clear"):
                    running_transcript = ""
                    await websocket.send_json({"ack": True})
                    continue
                if payload.get("finalize"):
                    await websocket.send_json(
                        {
                            "ack": True,
                            "finalized": True,
                            "text": running_transcript,
                        }
                    )
                continue

            if not audio_bytes:
                continue

            try:
                text = await asyncio.to_thread(transcribe_audio_bytes, recognizer, audio_bytes)
                if text:
                    running_transcript = merge_with_overlap(running_transcript, text)
                    await websocket.send_json({"ack": True, "text": running_transcript})
                else:
                    await websocket.send_json({"ack": True})
            except sr.UnknownValueError:
                await websocket.send_json({"ack": True})
            except sr.RequestError:
                await websocket.send_json({"ack": True, "error": "Google API unavailable."})
            except Exception as exc:
                await websocket.send_json({"ack": True, "error": f"Processing failed: {exc}"})
    except WebSocketDisconnect:
        return


@app.post("/session")
async def create_session() -> JSONResponse:
    session_id = sessions.create_session()
    return JSONResponse({"session_id": session_id})


@app.post("/transcribe-chunk")
async def transcribe_chunk(request: Request) -> JSONResponse:
    session_id = request.query_params.get("session_id", "")
    queue = sessions.get_queue(session_id)
    if not queue:
        return JSONResponse({"error": "Invalid session_id."}, status_code=400)

    data = await request.body()
    if not data:
        return JSONResponse({"ok": True})

    try:
        text = transcribe_audio_bytes(recognizer, data)
        if text:
            merged = merge_with_overlap(sessions.get_transcript(session_id), text)
            sessions.set_transcript(session_id, merged)
            await queue.put(merged)
        return JSONResponse({"ok": True})
    except sr.UnknownValueError:
        return JSONResponse({"ok": True})
    except sr.RequestError:
        return JSONResponse({"error": "Google API unavailable."}, status_code=503)
    except Exception as exc:
        return JSONResponse({"error": f"Processing failed: {exc}"}, status_code=500)


@app.get("/stream/{session_id}")
async def stream(session_id: str) -> StreamingResponse:
    queue = sessions.get_queue(session_id)
    if not queue:
        return StreamingResponse(iter(["event: error\ndata: Invalid session\n\n"]), media_type="text/event-stream")

    async def event_generator():
        try:
            while True:
                try:
                    text = await asyncio.wait_for(queue.get(), timeout=15)
                    if text:
                        yield f"data: {text}\n\n"
                except asyncio.TimeoutError:
                    yield ": keep-alive\n\n"
        except asyncio.CancelledError:
            return
        finally:
            sessions.end_session(session_id)

    return StreamingResponse(event_generator(), media_type="text/event-stream")


def run() -> None:
    uvicorn.run("speech_app.api:app", host="0.0.0.0", port=8001, reload=True)
