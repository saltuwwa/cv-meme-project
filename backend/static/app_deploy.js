/**
 * Deploy frontend — webcam polling to POST /predict.
 */

(function () {
  'use strict';

  const POLL_INTERVAL_MS = 1000;
  const CONFIDENCE_THRESHOLD = 0.75;

  const webcam = document.getElementById('webcam');
  const captureCanvas = document.getElementById('capture-canvas');
  const cameraPlaceholder = document.getElementById('camera-placeholder');
  const cameraOnBtn = document.getElementById('camera-on-btn');
  const cameraOffBtn = document.getElementById('camera-off-btn');
  const placeholder = document.getElementById('placeholder');
  const placeholderMessage = document.getElementById('placeholder-message');
  const result = document.getElementById('result');
  const resultMeme = document.getElementById('result-meme');
  const resultClass = document.getElementById('result-class');
  const resultConfidence = document.getElementById('result-confidence');

  let stream = null;
  let pollTimerId = null;
  let isRequestInFlight = false;

  const LOW_CONF_MESSAGE = 'Make one of the 4 poses clearly.';
  const CAMERA_DENIED_HTML =
    '<span class="placeholder-text">We need camera access to run the demo. Please allow the camera in your browser settings and try again.</span>';

  async function turnCameraOn() {
    try {
      stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: 'user', width: { ideal: 640 }, height: { ideal: 480 } },
        audio: false,
      });
      webcam.srcObject = stream;
      cameraPlaceholder.classList.add('hidden');
      cameraOnBtn.disabled = true;
      cameraOffBtn.disabled = false;
      startPolling();
    } catch (err) {
      console.error('Camera error:', err);
      cameraPlaceholder.innerHTML = CAMERA_DENIED_HTML;
      cameraPlaceholder.classList.remove('hidden');
    }
  }

  function turnCameraOff() {
    stopPolling();
    if (stream) {
      stream.getTracks().forEach((t) => t.stop());
      stream = null;
    }
    webcam.srcObject = null;
    cameraPlaceholder.innerHTML =
      '<span class="placeholder-text">Click “Turn Camera On” to start</span>';
    cameraPlaceholder.classList.remove('hidden');
    cameraOnBtn.disabled = false;
    cameraOffBtn.disabled = true;
    showPlaceholder();
  }

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

  function captureFrame() {
    captureCanvas.width = webcam.videoWidth;
    captureCanvas.height = webcam.videoHeight;
    const ctx = captureCanvas.getContext('2d');
    ctx.drawImage(webcam, 0, 0);
    return new Promise((resolve) => {
      captureCanvas.toBlob(resolve, 'image/jpeg', 0.9);
    });
  }

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

  function handleResult(data) {
    if (data.confidence >= CONFIDENCE_THRESHOLD) {
      showMeme(data);
    } else {
      showPlaceholder();
    }
  }

  function showMeme(data) {
    placeholder.classList.add('hidden');
    resultMeme.innerHTML = `<img src="${data.meme_path}" alt="${data.class}" />`;
    resultClass.textContent = data.class;
    resultConfidence.textContent = `${(data.confidence * 100).toFixed(1)}% confident`;
    result.classList.remove('hidden');
  }

  function showPlaceholder() {
    result.classList.add('hidden');
    placeholderMessage.textContent = LOW_CONF_MESSAGE;
    placeholder.classList.remove('hidden');
  }

  function cleanup() {
    turnCameraOff();
  }

  cameraOnBtn.addEventListener('click', turnCameraOn);
  cameraOffBtn.addEventListener('click', turnCameraOff);
  window.addEventListener('beforeunload', cleanup);
})();
