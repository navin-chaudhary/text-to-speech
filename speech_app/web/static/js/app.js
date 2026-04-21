(() => {
  const startMicBtn = document.getElementById("startMicBtn");
  const stopMicBtn = document.getElementById("stopMicBtn");
  const clearBtn = document.getElementById("clearBtn");
  const copyBtn = document.getElementById("copyBtn");
  const fileInput = document.getElementById("audioFile");
  const fileLabel = document.getElementById("fileLabel");
  const sendBtn = document.getElementById("sendBtn");
  const status = document.getElementById("status");
  const statusDot = document.getElementById("statusDot");
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
    statusDot.className = "status-dot";
    const lower = message.toLowerCase();
    if (lower.includes("recording") && lower.includes("stream")) {
      statusDot.classList.add("recording");
    } else if (lower.includes("error") || lower.includes("failed") || lower.includes("unavailable")) {
      statusDot.classList.add("error");
    } else {
      statusDot.classList.add("idle");
    }
  }

  function syncCopyButton() {
    copyBtn.disabled = !transcriptText.trim();
  }

  function appendTranscript(text) {
    if (!text || !text.trim()) return;
    const clean = text.trim();
    if (clean === lastChunkText) return;
    transcriptText = clean;
    lastChunkText = clean;
    result.textContent = transcriptText;
    syncCopyButton();
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
        } catch {
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
      body: blob,
    });
    const payload = await response.json();
    if (!response.ok) {
      throw new Error(payload.error || "Transcription failed.");
    }
    return payload.text || "(empty transcription)";
  }

  fileInput.addEventListener("change", () => {
    const file = fileInput.files[0];
    fileLabel.textContent = file ? file.name : "Choose file";
  });

  copyBtn.addEventListener("click", async () => {
    const text = transcriptText.trim();
    if (!text) return;
    try {
      await navigator.clipboard.writeText(text);
      setStatus("Copied transcript to clipboard.");
    } catch {
      setStatus("Could not copy (clipboard permission denied).");
    }
  });

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

      document.body.classList.add("is-recording");
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
      await waitForInFlightAck(3000);
      await runFlush(true);
      await waitForInFlightAck(3000);
      requestFinalize();
      await waitForFinalizeAck(3500);
      setStatus("Recording stopped.");
    } catch (err) {
      setStatus("Request failed: " + err);
    } finally {
      document.body.classList.remove("is-recording");
      if (processorNode) processorNode.disconnect();
      if (sourceNode) sourceNode.disconnect();
      if (audioContext) await audioContext.close();
      if (mediaStream) mediaStream.getTracks().forEach((track) => track.stop());
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
    syncCopyButton();
    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify({ clear: true }));
    }
    finalizeAck = false;
    finalizeInFlight = false;
    setStatus("Transcript cleared.");
  });

  setStatus("Idle.");
})();
