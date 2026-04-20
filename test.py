import os
import tempfile
import asyncio
import json
from uuid import uuid4

import speech_recognition as sr
import uvicorn
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse

app = FastAPI(title="Browser Speech-to-Text Demo")
recognizer = sr.Recognizer()
session_queues: dict[str, asyncio.Queue] = {}
session_transcripts: dict[str, str] = {}


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


def transcribe_audio_bytes(audio_bytes: bytes) -> str:
    temp_path = ""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(audio_bytes)
            temp_path = tmp.name

        with sr.AudioFile(temp_path) as source:
            audio = recognizer.record(source)

        return recognizer.recognize_google(audio).strip()
    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)


@app.get("/", response_class=HTMLResponse)
async def index() -> str:
    return """
    <!doctype html>
    <html>
    <head>
      <meta charset="utf-8" />
      <title>Speech to Text</title>
      <style>
        body { font-family: sans-serif; max-width: 680px; margin: 2rem auto; }
        button { margin-left: 0.5rem; }
        pre { background: #f5f5f5; padding: 1rem; border-radius: 8px; white-space: pre-wrap; }
      </style>
    </head>
    <body>
      <h2>Speech to text (browser microphone)</h2>
      <p>Use your browser microphone, or upload an audio file. Mic mode uses adaptive WebSocket streaming (silence-based chunking).</p>
      <button id="startMicBtn">Start Mic</button>
      <button id="stopMicBtn" disabled>Stop + Transcribe</button>
      <button id="clearBtn">Clear</button>
      <hr />
      <input id="audioFile" type="file" accept="audio/*" />
      <button id="sendBtn">Transcribe</button>
      <h3>Status</h3>
      <pre id="status">Idle.</pre>
      <h3>Result</h3>
      <pre id="result"></pre>

      <script>
        const startMicBtn = document.getElementById("startMicBtn");
        const stopMicBtn = document.getElementById("stopMicBtn");
        const clearBtn = document.getElementById("clearBtn");
        const fileInput = document.getElementById("audioFile");
        const sendBtn = document.getElementById("sendBtn");
        const status = document.getElementById("status");
        const result = document.getElementById("result");
        let audioContext = null;
        let mediaStream = null;
        let sourceNode = null;
        let processorNode = null;
        let sampleRate = 16000;
        let ws = null;
        let segmentBuffers = [];
        let segmentSamples = 0;
        let trailingSilenceMs = 0;
        let hasSpeechInSegment = false;
        let transcriptText = "";
        let lastChunkText = "";
        let flushInProgress = false;
        let flushPending = false;
        let flushPendingForce = false;
        let chunkInFlight = false;
        let finalizeInFlight = false;
        let finalizeAck = false;
        const SILENCE_THRESHOLD = 0.015;
        const MIN_CHUNK_MS = 320;
        const SILENCE_FLUSH_MS = 220;
        const MAX_CHUNK_MS = 900;

        function setStatus(message) {
          status.textContent = message;
        }

        function appendTranscript(text) {
          if (!text || !text.trim()) return;
          const clean = text.trim();
          if (clean === lastChunkText) return;
          transcriptText = clean;
          lastChunkText = clean;
          result.textContent = transcriptText;
        }

        function convertFloatTo16BitPCM(float32Array) {
          const output = new Int16Array(float32Array.length);
          for (let i = 0; i < float32Array.length; i++) {
            const s = Math.max(-1, Math.min(1, float32Array[i]));
            output[i] = s < 0 ? s * 0x8000 : s * 0x7fff;
          }
          return output;
        }

        function encodeWav(float32Array, sr) {
          const pcm16 = convertFloatTo16BitPCM(float32Array);
          const dataSize = pcm16.length * 2;
          const buffer = new ArrayBuffer(44 + dataSize);
          const view = new DataView(buffer);

          function writeString(offset, text) {
            for (let i = 0; i < text.length; i++) {
              view.setUint8(offset + i, text.charCodeAt(i));
            }
          }

          writeString(0, "RIFF");
          view.setUint32(4, 36 + dataSize, true);
          writeString(8, "WAVE");
          writeString(12, "fmt ");
          view.setUint32(16, 16, true);
          view.setUint16(20, 1, true);
          view.setUint16(22, 1, true);
          view.setUint32(24, sr, true);
          view.setUint32(28, sr * 2, true);
          view.setUint16(32, 2, true);
          view.setUint16(34, 16, true);
          writeString(36, "data");
          view.setUint32(40, dataSize, true);

          let offset = 44;
          for (let i = 0; i < pcm16.length; i++, offset += 2) {
            view.setInt16(offset, pcm16[i], true);
          }

          return new Blob([buffer], { type: "audio/wav" });
        }

        function calculateRms(float32Array) {
          let sum = 0;
          for (let i = 0; i < float32Array.length; i++) {
            sum += float32Array[i] * float32Array[i];
          }
          return Math.sqrt(sum / float32Array.length);
        }

        function mergeSegmentBuffers() {
          if (!segmentBuffers.length || segmentSamples === 0) {
            return null;
          }
          const merged = new Float32Array(segmentSamples);
          let offset = 0;
          for (const chunk of segmentBuffers) {
            merged.set(chunk, offset);
            offset += chunk.length;
          }
          return merged;
        }

        function resetSegment() {
          segmentBuffers = [];
          segmentSamples = 0;
          trailingSilenceMs = 0;
          hasSpeechInSegment = false;
        }

        function getWebSocketUrl() {
          const scheme = location.protocol === "https:" ? "wss" : "ws";
          return `${scheme}://${location.host}/ws/transcribe`;
        }

        async function connectWebSocket() {
          return new Promise((resolve, reject) => {
            const socket = new WebSocket(getWebSocketUrl());
            const timeout = setTimeout(() => {
              socket.close();
              reject(new Error("WebSocket connect timeout."));
            }, 4000);

            socket.onopen = () => {
              clearTimeout(timeout);
              resolve(socket);
            };
            socket.onmessage = (event) => {
              try {
                const payload = JSON.parse(event.data);
                if (payload.ack) {
                  chunkInFlight = false;
                }
                if (payload.finalized) {
                  finalizeAck = true;
                  finalizeInFlight = false;
                }
                if (payload.text) {
                  appendTranscript(payload.text);
                }
                if (payload.error) {
                  setStatus(`Server error: ${payload.error}`);
                }
              } catch (_) {
                // Ignore malformed messages.
              }
            };
            socket.onerror = () => {
              setStatus("Live stream error.");
            };
            socket.onclose = () => {
              if (startMicBtn.disabled) {
                setStatus("Live stream disconnected.");
              }
            };
          });
        }

        async function flushSegment(forceFlush = false) {
          if (!ws || ws.readyState !== WebSocket.OPEN) {
            return;
          }
          if (chunkInFlight) {
            return;
          }
          const chunkMs = segmentSamples > 0 ? (segmentSamples * 1000) / sampleRate : 0;
          if (!forceFlush && chunkMs < MIN_CHUNK_MS) {
            return;
          }

          const merged = mergeSegmentBuffers();
          if (!merged) return;

          const wavBlob = encodeWav(merged, sampleRate);
          chunkInFlight = true;
          ws.send(wavBlob);
          resetSegment();
        }

        async function runFlush(forceFlush = false) {
          if (flushInProgress) {
            flushPending = true;
            flushPendingForce = flushPendingForce || forceFlush;
            return;
          }
          flushInProgress = true;
          try {
            await flushSegment(forceFlush);
          } finally {
            flushInProgress = false;
            if (flushPending) {
              const nextForce = flushPendingForce;
              flushPending = false;
              flushPendingForce = false;
              await runFlush(nextForce);
            }
          }
        }

        async function waitForFlushCompletion() {
          while (flushInProgress) {
            await new Promise((resolve) => setTimeout(resolve, 10));
          }
        }

        async function waitForInFlightAck(timeoutMs = 1800) {
          const start = Date.now();
          while (chunkInFlight && Date.now() - start < timeoutMs) {
            await new Promise((resolve) => setTimeout(resolve, 20));
          }
        }

        function requestFinalize() {
          if (!ws || ws.readyState !== WebSocket.OPEN) return;
          finalizeAck = false;
          finalizeInFlight = true;
          ws.send(JSON.stringify({ finalize: true }));
        }

        async function waitForFinalizeAck(timeoutMs = 3500) {
          const start = Date.now();
          while (finalizeInFlight && !finalizeAck && Date.now() - start < timeoutMs) {
            await new Promise((resolve) => setTimeout(resolve, 20));
          }
        }

        async function sendAudioBlob(blob) {
          const response = await fetch("/transcribe", {
            method: "POST",
            headers: { "Content-Type": "audio/wav" },
            body: blob
          });
          const payload = await response.json();
          if (!response.ok) {
            throw new Error(payload.error || "Transcription failed.");
          }
          return payload.text || "(empty transcription)";
        }

        startMicBtn.addEventListener("click", async () => {
          try {
            setStatus("Requesting microphone access...");
            resetSegment();
            finalizeAck = false;
            finalizeInFlight = false;
            ws = await connectWebSocket();
            mediaStream = await navigator.mediaDevices.getUserMedia({ audio: true });
            audioContext = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: 16000 });
            sampleRate = audioContext.sampleRate;
            sourceNode = audioContext.createMediaStreamSource(mediaStream);
            processorNode = audioContext.createScriptProcessor(4096, 1, 1);
            sourceNode.connect(processorNode);
            processorNode.connect(audioContext.destination);
            processorNode.onaudioprocess = (event) => {
              if (!ws || ws.readyState !== WebSocket.OPEN) return;
              const channelData = event.inputBuffer.getChannelData(0);
              const frame = new Float32Array(channelData);
              const frameMs = (frame.length * 1000) / sampleRate;
              const rms = calculateRms(frame);

              segmentBuffers.push(frame);
              segmentSamples += frame.length;

              if (rms >= SILENCE_THRESHOLD) {
                hasSpeechInSegment = true;
                trailingSilenceMs = 0;
              } else {
                trailingSilenceMs += frameMs;
              }

              const chunkMs = (segmentSamples * 1000) / sampleRate;
              const shouldFlushBySilence = hasSpeechInSegment && trailingSilenceMs >= SILENCE_FLUSH_MS && chunkMs >= MIN_CHUNK_MS;
              const shouldFlushByMaxLatency = chunkMs >= MAX_CHUNK_MS;
              if (shouldFlushBySilence || shouldFlushByMaxLatency) {
                void runFlush(false);
              }
            };

            startMicBtn.disabled = true;
            stopMicBtn.disabled = false;
            setStatus("Recording with adaptive streaming...");
          } catch (err) {
            setStatus("Microphone error: " + err);
          }
        });

        stopMicBtn.addEventListener("click", async () => {
          try {
            stopMicBtn.disabled = true;
            setStatus("Preparing audio...");
            if (processorNode) {
              processorNode.onaudioprocess = null;
            }
            await waitForFlushCompletion();
            // Ensure only one outstanding chunk at a time before final flush.
            await waitForInFlightAck(3000);
            await runFlush(true);
            await waitForInFlightAck(3000);
            requestFinalize();
            await waitForFinalizeAck(3500);
            setStatus("Recording stopped.");
          } catch (err) {
            setStatus("Request failed: " + err);
          } finally {
            if (processorNode) processorNode.disconnect();
            if (sourceNode) sourceNode.disconnect();
            if (audioContext) await audioContext.close();
            if (mediaStream) mediaStream.getTracks().forEach(track => track.stop());
            processorNode = null;
            sourceNode = null;
            audioContext = null;
            mediaStream = null;
            if (ws && ws.readyState === WebSocket.OPEN) {
              ws.close();
            }
            ws = null;
            chunkInFlight = false;
            finalizeAck = false;
            finalizeInFlight = false;
            resetSegment();
            startMicBtn.disabled = false;
            stopMicBtn.disabled = true;
          }
        });

        sendBtn.addEventListener("click", async () => {
          const file = fileInput.files[0];
          if (!file) {
            setStatus("Please choose an audio file first.");
            return;
          }
          setStatus("Transcribing...");
          try {
            const text = await sendAudioBlob(file);
            appendTranscript(text);
            setStatus("Done.");
          } catch (err) {
            setStatus("Request failed: " + err);
          }
        });

        clearBtn.addEventListener("click", () => {
          transcriptText = "";
          lastChunkText = "";
          result.textContent = "";
          if (ws && ws.readyState === WebSocket.OPEN) {
            ws.send(JSON.stringify({ clear: true }));
          }
          finalizeAck = false;
          finalizeInFlight = false;
          setStatus("Transcript cleared.");
        });
      </script>
    </body>
    </html>
    """


@app.post("/transcribe")
async def transcribe(request: Request) -> JSONResponse:
    data = await request.body()
    if not data:
        return JSONResponse({"error": "No audio data received."}, status_code=400)

    temp_path = ""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(data)
            temp_path = tmp.name

        with sr.AudioFile(temp_path) as source:
            audio = recognizer.record(source)

        text = recognizer.recognize_google(audio)
        return JSONResponse({"text": text})
    except sr.UnknownValueError:
        return JSONResponse({"error": "Could not understand audio."}, status_code=422)
    except sr.RequestError:
        return JSONResponse({"error": "Google API unavailable."}, status_code=503)
    except Exception as exc:
        return JSONResponse({"error": f"Processing failed: {exc}"}, status_code=500)
    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)


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
                except Exception:
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
                # Offload blocking decoding + network call off event loop.
                text = await asyncio.to_thread(transcribe_audio_bytes, audio_bytes)
                if text:
                    running_transcript = merge_with_overlap(running_transcript, text)
                    await websocket.send_json({"ack": True, "text": running_transcript})
                else:
                    await websocket.send_json({"ack": True})
            except sr.UnknownValueError:
                # No recognizable speech in this chunk; ignore.
                await websocket.send_json({"ack": True})
            except sr.RequestError:
                await websocket.send_json({"ack": True, "error": "Google API unavailable."})
            except Exception as exc:
                await websocket.send_json({"ack": True, "error": f"Processing failed: {exc}"})
    except WebSocketDisconnect:
        return


@app.post("/session")
async def create_session() -> JSONResponse:
    session_id = str(uuid4())
    session_queues[session_id] = asyncio.Queue()
    session_transcripts[session_id] = ""
    return JSONResponse({"session_id": session_id})


@app.post("/transcribe-chunk")
async def transcribe_chunk(request: Request) -> JSONResponse:
    session_id = request.query_params.get("session_id", "")
    queue = session_queues.get(session_id)
    if not queue:
        return JSONResponse({"error": "Invalid session_id."}, status_code=400)

    data = await request.body()
    if not data:
        return JSONResponse({"ok": True})

    temp_path = ""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(data)
            temp_path = tmp.name

        with sr.AudioFile(temp_path) as source:
            audio = recognizer.record(source)

        text = recognizer.recognize_google(audio).strip()
        if text:
            merged_text = merge_with_overlap(session_transcripts.get(session_id, ""), text)
            session_transcripts[session_id] = merged_text
            await queue.put(merged_text)
        return JSONResponse({"ok": True})
    except sr.UnknownValueError:
        return JSONResponse({"ok": True})
    except sr.RequestError:
        return JSONResponse({"error": "Google API unavailable."}, status_code=503)
    except Exception as exc:
        return JSONResponse({"error": f"Processing failed: {exc}"}, status_code=500)
    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)


@app.get("/stream/{session_id}")
async def stream(session_id: str) -> StreamingResponse:
    queue = session_queues.get(session_id)
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
            session_queues.pop(session_id, None)
            session_transcripts.pop(session_id, None)

    return StreamingResponse(event_generator(), media_type="text/event-stream")


if __name__ == "__main__":
    uvicorn.run("test:app", host="0.0.0.0", port=8001, reload=True)
