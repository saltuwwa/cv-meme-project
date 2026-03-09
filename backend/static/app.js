/**
 * app.js - Qazaq Meme Vision
 * Real-time meme mirror: auto-capture frames, poll /predict, show meme when confident.
 */

(function () {
  'use strict';

  // Config
  const POLL_INTERVAL_MS = 1000;        // Send frame every ~1 second
  const CONFIDENCE_THRESHOLD = 0.75;     // Show meme only if confidence >= 75%

  // DOM elements
  const webcam = document.getElementById('webcam');
  const captureCanvas = document.getElementById('capture-canvas');
  const cameraPlaceholder = document.getElementById('camera-placeholder');
  const cameraOnBtn = document.getElementById('camera-on-btn');
  const cameraOffBtn = document.getElementById('camera-off-btn');
  const placeholder = document.getElementById('placeholder');
  const result = document.getElementById('result');
  const resultMeme = document.getElementById('result-meme');
  const resultClass = document.getElementById('result-class');
  const resultConfidence = document.getElementById('result-confidence');

  let stream = null;
  let pollTimerId = null;
  let isRequestInFlight = false;

  /**
   * Turn camera on: request stream, show video, start polling.
   */
  async function turnCameraOn() {
    try {
      stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: 'user', width: { ideal: 640 }, height: { ideal: 480 } },
        audio: false
      });
      webcam.srcObject = stream;
      cameraPlaceholder.classList.add('hidden');
      cameraOnBtn.disabled = true;
      cameraOffBtn.disabled = false;
      startPolling();
    } catch (err) {
      console.error('Camera error:', err);
      cameraPlaceholder.textContent = 'Camera access denied';
      cameraPlaceholder.classList.remove('hidden');
    }
  }

  /**
   * Turn camera off: stop stream, stop polling, reset UI.
   */
  function turnCameraOff() {
    stopPolling();
    if (stream) {
      stream.getTracks().forEach((t) => t.stop());
      stream = null;
    }
    webcam.srcObject = null;
    cameraPlaceholder.textContent = 'Click "Turn Camera On" to start';
    cameraPlaceholder.classList.remove('hidden');
    cameraOnBtn.disabled = false;
    cameraOffBtn.disabled = true;
    showPlaceholder();
  }

  /**
   * Start periodic frame capture and prediction.
   * Sends a frame every POLL_INTERVAL_MS. Does not overlap requests.
   */
  function startPolling() {
    stopPolling();
    pollTimerId = setInterval(pollOnce, POLL_INTERVAL_MS);
  }

  function stopPolling() {
    if (pollTimerId) {
      clearInterval(pollTimerId);
      pollTimerId = null;
    }
  }

  /**
   * Capture one frame and send to /predict.
   * Skips if a request is already in flight (prevents overlapping).
   */
  async function pollOnce() {
    if (!stream || webcam.readyState !== 4) return;
    if (isRequestInFlight) return;

    isRequestInFlight = true;
    try {
      const blob = await captureFrame();
      const data = await predict(blob);
      handleResult(data);
    } catch (err) {
      console.error('Predict error:', err);
    } finally {
      isRequestInFlight = false;
    }
  }

  /**
   * Capture current webcam frame as JPEG blob.
   */
  function captureFrame() {
    captureCanvas.width = webcam.videoWidth;
    captureCanvas.height = webcam.videoHeight;
    const ctx = captureCanvas.getContext('2d');
    ctx.drawImage(webcam, 0, 0);
    return new Promise((resolve) => {
      captureCanvas.toBlob(resolve, 'image/jpeg', 0.9);
    });
  }

  /**
   * POST image to /predict, return { class, confidence, meme_path }.
   */
  async function predict(blob) {
    const formData = new FormData();
    formData.append('file', blob, 'capture.jpg');
    const response = await fetch('/predict', { method: 'POST', body: formData });
    if (!response.ok) {
      const err = await response.json().catch(() => ({ detail: response.statusText }));
      throw new Error(err.detail || 'Prediction failed');
    }
    return response.json();
  }

  /**
   * Handle prediction result.
   * If confidence >= threshold: show meme.
   * Else: show calm placeholder.
   */
  function handleResult(data) {
    if (data.confidence >= CONFIDENCE_THRESHOLD) {
      showMeme(data);
    } else {
      showPlaceholder();
    }
  }

  /**
   * Show meme result with smooth animation.
   */
  function showMeme(data) {
    placeholder.classList.add('hidden');
    resultMeme.innerHTML = `<img src="${data.meme_path}" alt="${data.class}" />`;
    resultClass.textContent = data.class;
    resultConfidence.textContent = `${(data.confidence * 100).toFixed(1)}% confident`;
    result.classList.remove('hidden');
  }

  /**
   * Show calm placeholder (no confident prediction).
   */
  function showPlaceholder() {
    result.classList.add('hidden');
    placeholder.classList.remove('hidden');
  }

  /**
   * Cleanup on page unload.
   */
  function cleanup() {
    turnCameraOff();
  }

  // Events
  cameraOnBtn.addEventListener('click', turnCameraOn);
  cameraOffBtn.addEventListener('click', turnCameraOff);
  window.addEventListener('beforeunload', cleanup);
})();
