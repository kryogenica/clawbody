#!/usr/bin/env python3
"""Always-on wake-word listener for Reachy Mini.

Listens to the robot microphone continuously - even while the robot is "off"
(asleep) - and, when it hears the wake phrase, powers the robot ON by calling
the control center's existing REST API (POST /api/power/on).

Why this works while the robot is off:
  * The reachy-mini-daemon process keeps the USB mic (hw:0,0) open at all times,
    but exposes it through an ALSA *dsnoop* PCM (`reachymini_audio_src`, defined
    in ~/.asoundrc). dsnoop allows multiple processes to capture the same device
    simultaneously, so this listener can read the mic concurrently with the
    daemon without any device-busy conflict.
  * Audio is captured by piping `arecord -D reachymini_audio_src` (no PortAudio /
    sounddevice dependency), downmixed to mono 16 kHz, and fed to openWakeWord.

The robot is only powered on when it is currently OFF; if it is already running
the detection is ignored (the conversation app owns the mic interaction then).

Runs as a systemd service. openWakeWord + onnxruntime live in a dedicated venv
(scripts/scheduler/wake-venv) so the stdlib-only control server stays clean.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np
import openwakeword
from openwakeword.model import Model

# --------------------------------------------------------------------------- #
# Config (overridable via environment in the systemd unit)
# --------------------------------------------------------------------------- #
CONTROL_URL = os.getenv("REACHY_CONTROL_URL", "http://127.0.0.1:1991")
CAPTURE_DEVICE = os.getenv("REACHY_WAKE_DEVICE", "reachymini_audio_src")
SAMPLE_RATE = 16000
CHANNELS = int(os.getenv("REACHY_WAKE_CHANNELS", "2"))  # dsnoop src is stereo
CHUNK = 1280  # 80 ms @ 16 kHz - openWakeWord's expected frame size
THRESHOLD = float(os.getenv("REACHY_WAKE_THRESHOLD", "0.5"))
COOLDOWN_S = float(os.getenv("REACHY_WAKE_COOLDOWN", "10"))
STATUS_TIMEOUT = 4.0

_BUNDLED = Path(openwakeword.__file__).resolve().parent / "resources" / "models"
# Default to a bundled pretrained model for bring-up; point REACHY_WAKE_MODEL at
# the custom "Reachy Wake up" model once it is trained.
DEFAULT_MODEL = str(_BUNDLED / "hey_jarvis_v0.1.onnx")
MODEL_PATH = os.getenv("REACHY_WAKE_MODEL", DEFAULT_MODEL)


def log(msg: str) -> None:
    print(f"[wake] {time.strftime('%H:%M:%S')} {msg}", flush=True)


# --------------------------------------------------------------------------- #
# Control-center API
# --------------------------------------------------------------------------- #
def robot_is_on() -> bool | None:
    """True if the daemon backend is up, False if off, None if unknown."""
    try:
        with urllib.request.urlopen(f"{CONTROL_URL}/api/status", timeout=STATUS_TIMEOUT) as r:
            data = json.loads(r.read().decode())
    except (urllib.error.URLError, OSError, json.JSONDecodeError) as exc:
        log(f"status check failed: {exc}")
        return None
    if data.get("busy"):
        return True  # a power action is mid-flight; treat as on/busy, don't retrigger
    return data.get("daemon_state") in ("running", "starting")


def power_on() -> None:
    body = json.dumps({}).encode()
    req = urllib.request.Request(
        f"{CONTROL_URL}/api/power/on", data=body, method="POST",
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=STATUS_TIMEOUT) as r:
            log(f"power_on -> {r.status} {r.read().decode().strip()}")
    except (urllib.error.URLError, OSError) as exc:
        log(f"power_on request failed: {exc}")


# --------------------------------------------------------------------------- #
# Audio capture (arecord -> raw stdout)
# --------------------------------------------------------------------------- #
def open_capture() -> subprocess.Popen:
    cmd = [
        "arecord", "-q", "-D", CAPTURE_DEVICE,
        "-f", "S16_LE", "-c", str(CHANNELS), "-r", str(SAMPLE_RATE),
        "-t", "raw",
    ]
    log(f"starting capture: {' '.join(cmd)}")
    return subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)


def read_chunk(proc: subprocess.Popen) -> np.ndarray | None:
    """Read one CHUNK of mono int16 samples, or None on stream end."""
    want = CHUNK * CHANNELS * 2  # bytes
    buf = proc.stdout.read(want)
    if not buf or len(buf) < want:
        return None
    samples = np.frombuffer(buf, dtype=np.int16)
    if CHANNELS > 1:
        samples = samples.reshape(-1, CHANNELS).mean(axis=1).astype(np.int16)
    return samples


# --------------------------------------------------------------------------- #
# Main loop
# --------------------------------------------------------------------------- #
def main() -> None:
    if not Path(MODEL_PATH).is_file():
        log(f"FATAL: model not found: {MODEL_PATH}")
        sys.exit(1)

    log(f"loading model: {MODEL_PATH} (threshold={THRESHOLD})")
    model = Model(wakeword_model_paths=[MODEL_PATH])
    keys = list(model.models.keys())
    log(f"loaded wakeword(s): {keys}; listening on '{CAPTURE_DEVICE}'")

    last_fire = 0.0
    proc = open_capture()

    while True:
        chunk = read_chunk(proc)
        if chunk is None:
            log("capture stream ended; restarting in 2s")
            try:
                proc.kill()
            except Exception:  # noqa: BLE001
                pass
            time.sleep(2)
            proc = open_capture()
            model.reset()
            continue

        scores = model.predict(chunk)
        score = max(scores.values()) if scores else 0.0

        if score < THRESHOLD:
            continue

        now = time.time()
        if now - last_fire < COOLDOWN_S:
            continue
        last_fire = now

        log(f"WAKE detected (score={score:.3f})")
        on = robot_is_on()
        if on is True:
            log("robot already on/busy; ignoring")
        elif on is False:
            log("robot is off -> powering on")
            power_on()
        else:
            log("robot state unknown; skipping power-on this time")


if __name__ == "__main__":
    main()
