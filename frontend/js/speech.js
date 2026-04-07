// frontend/js/speech.js
// v9 — Silero VAD + Raw PCM WAV (cross-browser, ECAPA-optimised)
//
// UPGRADE FROM v8:
//   v8 used ScriptProcessorNode to capture raw PCM — fixing the Edge-only
//   codec bug. v9 keeps raw PCM WAV but adds Silero VAD (Voice Activity
//   Detection) via the @ricky0123/vad-web library already loaded in the page.
//
// WHY VAD IMPROVES ECAPA ACCURACY:
//   ECAPA-TDNN computes a speaker embedding over the ENTIRE audio buffer.
//   If the buffer contains silence, background noise, or breath sounds at
//   the start/end, those non-speech frames dilute the embedding and push
//   cosine similarity down — causing false rejects. Silero VAD is a neural
//   network (runs locally in ONNX Runtime) that detects exactly which frames
//   contain real speech. Only those frames are sent to the backend, giving
//   ECAPA a clean, speech-only signal to work with.
//
// HOW IT WORKS:
//   1. Mic opens → Silero VAD starts listening in real time
//   2. VAD detects speech start → recording indicator turns green
//   3. VAD detects speech end (300ms silence) → automatically stops,
//      collects only the speech frames, encodes to WAV, sends to backend
//   4. Manual Stop button still available as fallback (user can click to
//      force-stop before the silence timeout fires)
//   5. Auto-stop watchdog still present (8s max) as safety net
//
// NO STOP BUTTON REQUIRED in normal use — VAD auto-stops when you finish
// speaking. Stop button is shown but acts as a manual override.
//
// Dependencies (already loaded in login.html and enroll.html):
//   - onnxruntime-web  (runs the Silero ONNX model)
//   - @ricky0123/vad-web  (Silero VAD wrapper)
//
// Works on: Chrome, Edge, Firefox, Safari, Opera, iOS Safari, Android Chrome.
//
// Integration contract (unchanged from v4-v8):
//   - Global startRecording() called by record-btn onclick
//   - Calls onVoiceRecorded(featureDict)     during enrollment
//   - Calls onVoiceAuthComplete(featureDict) during login
//   - DOM ids: record-btn, voice-status, diag-text, recording-indicator

const SpeechCapture = {
    vad:             null,      // Silero VAD instance
    stream:          null,
    speechFrames:    [],        // Float32Array chunks from VAD (speech only)
    isRecording:     false,
    currentAttempt:  1,
    maxAttempts:     3,
    _watchdogTimer:  null,
    _vadStarted:     false,

    TARGET_SAMPLE_RATE: 16000,  // ECAPA-TDNN expects 16 kHz
    MAX_DURATION_MS:    10000,  // auto-stop watchdog (10s)
    MIN_SPEECH_MS:      800,    // reject if less than 0.8s of actual speech
    NOISE_THRESHOLD:    -25,    // dBFS for noise floor warning

    // ── DOM helpers ───────────────────────────────────────────────────────
    _setStatus(text, color) {
        const el = document.getElementById("voice-status");
        if (!el) return;
        el.textContent = text;
        el.className   = `text-center text-sm mb-4 ${color}`;
    },

    _setDiag(text) {
        const el = document.getElementById("diag-text");
        if (el) el.textContent = text;
    },

    _setIndicator(active, color) {
        const circle = document.getElementById("mic-circle");
        const svg    = document.getElementById("mic-svg");
        const ring1  = document.getElementById("mic-ring-1");
        const ring2  = document.getElementById("mic-ring-2");
        if (!circle) return;

        if (color === "#10b981") {
            // Speech detected — green, rings pulse
            circle.style.background = "#10b981";
            if (svg) svg.setAttribute("stroke", "#ffffff");
            if (ring1) { ring1.style.opacity = "0.6"; ring1.style.transform = "scale(1)"; }
            if (ring2) { ring2.style.opacity = "0.3"; ring2.style.transform = "scale(1)"; }
        } else if (color === "#f59e0b" || (!active && !color)) {
            // Waiting / idle — yellow, no rings
            circle.style.background = "#374151";
            if (svg) svg.setAttribute("stroke", "#f59e0b");
            if (ring1) { ring1.style.opacity = "0"; ring1.style.transform = "scale(0.75)"; }
            if (ring2) { ring2.style.opacity = "0"; ring2.style.transform = "scale(0.75)"; }
        } else {
            // Idle / reset
            circle.style.background = "#374151";
            if (svg) svg.setAttribute("stroke", "#9ca3af");
            if (ring1) { ring1.style.opacity = "0"; ring1.style.transform = "scale(0.75)"; }
            if (ring2) { ring2.style.opacity = "0"; ring2.style.transform = "scale(0.75)"; }
        }
    },

    _showStopBtn() {
        // Mic circle becomes a stop indicator — red background
        const circle = document.getElementById("mic-circle");
        const svg    = document.getElementById("mic-svg");
        if (circle) {
            circle.style.background = "#dc2626";
            circle.onclick = () => this.stopRecording();
        }
        if (svg) svg.setAttribute("stroke", "#ffffff");

        const btn = document.getElementById("record-btn");
        if (btn) {
            btn.disabled    = false;
            btn.textContent = "⏹ Stop Recording";
            btn.onclick     = () => this.stopRecording();
        }
        const tryAgain = document.getElementById("try-again-btn");
        if (tryAgain) tryAgain.disabled = true;
    },

    _showStartBtn(label) {
        // Mic circle resets to idle
        const circle = document.getElementById("mic-circle");
        const svg    = document.getElementById("mic-svg");
        if (circle) {
            circle.style.background = "#374151";
            circle.onclick = startRecording;
        }
        if (svg) svg.setAttribute("stroke", "#9ca3af");

        const btn = document.getElementById("record-btn");
        if (btn) {
            btn.disabled    = false;
            btn.textContent = label || "🎤 Start Recording";
            btn.onclick     = startRecording;
        }
        const tryAgain = document.getElementById("try-again-btn");
        if (tryAgain) tryAgain.disabled = false;
    },

    // ── Resample Float32 PCM srcRate → dstRate (linear interpolation) ─────
    _resample(samples, srcRate, dstRate) {
        if (srcRate === dstRate) return samples;
        const ratio  = srcRate / dstRate;
        const length = Math.round(samples.length / ratio);
        const result = new Float32Array(length);
        for (let i = 0; i < length; i++) {
            const pos  = i * ratio;
            const idx  = Math.floor(pos);
            const frac = pos - idx;
            result[i]  = (samples[idx] || 0) + frac * ((samples[idx + 1] || 0) - (samples[idx] || 0));
        }
        return result;
    },

    // ── Auto-gain on raw PCM samples ──────────────────────────────────────
    _autoGain(samples) {
        const rms = Math.sqrt(samples.reduce((s, x) => s + x * x, 0) / samples.length);
        if (rms < 0.001) return samples;
        const targetRms = 0.08;
        const gain      = Math.min(targetRms / rms, 10.0);
        if (rms >= targetRms) return samples;
        console.log(`[SpeechCapture] Auto-gain x${gain.toFixed(2)} (RMS ${rms.toFixed(4)})`);
        const out = new Float32Array(samples.length);
        for (let i = 0; i < samples.length; i++) {
            out[i] = Math.max(-1, Math.min(1, samples[i] * gain));
        }
        return out;
    },

    // ── Float32 mono PCM → 16-bit WAV Blob ───────────────────────────────
    _float32ToWav(samples, sampleRate) {
        const dataSize = samples.length * 2;
        const buf = new ArrayBuffer(44 + dataSize);
        const v   = new DataView(buf);
        const ws  = (o, s) => { for (let i = 0; i < s.length; i++) v.setUint8(o + i, s.charCodeAt(i)); };
        ws(0, "RIFF"); v.setUint32(4, 36 + dataSize, true);
        ws(8, "WAVE"); ws(12, "fmt ");
        v.setUint32(16, 16, true); v.setUint16(20, 1, true); v.setUint16(22, 1, true);
        v.setUint32(24, sampleRate, true); v.setUint32(28, sampleRate * 2, true);
        v.setUint16(32, 2, true); v.setUint16(34, 16, true);
        ws(36, "data"); v.setUint32(40, dataSize, true);
        let off = 44;
        for (let i = 0; i < samples.length; i++, off += 2) {
            const s = Math.max(-1, Math.min(1, samples[i]));
            v.setInt16(off, s < 0 ? s * 0x8000 : s * 0x7FFF, true);
        }
        return new Blob([buf], { type: "audio/wav" });
    },

    _blobToBase64(blob) {
        return new Promise((resolve, reject) => {
            const r     = new FileReader();
            r.onloadend = () => resolve(r.result.split(",")[1]);
            r.onerror   = reject;
            r.readAsDataURL(blob);
        });
    },

    // ── Main entry point ──────────────────────────────────────────────────
    async startRecording() {
        try {
            this._setStatus("⏳ Starting voice detection…", "text-yellow-400");
            this._setDiag("");
            this.speechFrames = [];
            this._vadStarted  = false;

            // Check VAD library is available
            if (typeof vad === "undefined" || !vad.MicVAD) {
                throw new Error("VAD library not loaded. Check your script tags.");
            }

            // Request mic permission early so user sees the browser prompt
            this.stream = await navigator.mediaDevices.getUserMedia({
                audio: {
                    channelCount:          1,
                    sampleRate:            { ideal: this.TARGET_SAMPLE_RATE },
                    echoCancellation:      true,
                    noiseSuppression:      true,
                    autoGainControl:       false,
                    googNoiseSuppression:  true,
                    googNoiseSuppression2: true,
                    googHighpassFilter:    true,
                }
            });

            this._setStatus("🧠 Loading voice detector…", "text-yellow-400");

            // Initialise Silero VAD
            // MicVAD handles AudioContext + ONNX model internally.
            // onFrameProcessed fires for every 30ms audio frame.
            // onSpeechStart / onSpeechEnd bracket real speech segments.
            this.vad = await vad.MicVAD.new({
                stream: this.stream,
                positiveSpeechThreshold: 0.85,   // confidence to call a frame "speech"
                negativeSpeechThreshold: 0.20,   // confidence to call a frame "silence"
                minSpeechFrames:         5,       // ignore very short blips (< ~150ms)
                preSpeechPadFrames:      8,       // keep 240ms before speech starts
                redemptionFrames:        10,      // wait 300ms silence before ending

                onSpeechStart: () => {
                    console.log("[VAD] Speech started");
                    this._vadStarted = true;
                    this._setStatus("🔴 Speaking detected — keep going…", "text-green-400");
                    this._setIndicator(true, "#10b981");  // green = speech detected
                },

                onFrameProcessed: (probabilities) => {
                    // Pulse mic ring intensity based on speech probability
                    const ring1 = document.getElementById("mic-ring-1");
                    const pct   = probabilities.isSpeech;
                    if (ring1 && SpeechCapture.isRecording) {
                        ring1.style.opacity = (pct * 0.7).toFixed(2);
                    }
                },

                onSpeechEnd: (audioFloat32) => {
                    // audioFloat32 is already 16kHz Float32Array from Silero VAD
                    // containing ONLY the speech segment — clean and trimmed.
                    console.log(`[VAD] Speech ended — ${audioFloat32.length} samples (${(audioFloat32.length / 16000).toFixed(2)}s)`);
                    this.speechFrames.push(audioFloat32);

                    const totalSamples = this.speechFrames.reduce((s, f) => s + f.length, 0);
                    const totalMs      = (totalSamples / this.TARGET_SAMPLE_RATE) * 1000;

                    if (totalMs >= this.MIN_SPEECH_MS) {
                        // Enough speech captured — process immediately
                        this._setStatus("✅ Voice captured! Processing…", "text-green-400");
                        this.stopRecording();
                    } else {
                        // Not enough yet — keep listening
                        this._setStatus(
                            `Got ${(totalMs / 1000).toFixed(1)}s — say the full phrase…`,
                            "text-yellow-400"
                        );
                        this._setIndicator(false);
                    }
                },
            });

            this.isRecording = true;
            this.vad.start();

            // Watchdog — force stop after MAX_DURATION_MS
            this._watchdogTimer = setTimeout(() => {
                if (this.isRecording) {
                    console.log("[SpeechCapture] Watchdog triggered");
                    this.stopRecording();
                }
            }, this.MAX_DURATION_MS);

            this._setStatus("🎤 Listening… speak your phrase", "text-blue-400");
            this._setIndicator(false, "#f59e0b");  // yellow = waiting for speech
            this._setDiag("Speak clearly — VAD will detect when you start and stop");
            this._showStopBtn();

            return true;

        } catch (err) {
            console.error("[SpeechCapture] Init error:", err);

            // Fallback: if VAD fails to load, tell the user clearly
            const msg = err.name === "NotAllowedError"
                ? "❌ Microphone access denied. Allow microphone and try again."
                : `❌ Could not start voice detector: ${err.message}`;
            this._setStatus(msg, "text-red-400");
            this._showStartBtn("🎤 Try Again");
            return false;
        }
    },

    // ── Stop recording (called by Stop button, watchdog, or VAD auto-stop) ─
    stopRecording() {
        if (this._watchdogTimer) {
            clearTimeout(this._watchdogTimer);
            this._watchdogTimer = null;
        }
        if (!this.isRecording) return;
        this.isRecording = false;

        this._setStatus("⚙️ Processing captured voice…", "text-yellow-400");
        this._setDiag("");

        // Reset mic visual
        this._setIndicator(false, null);

        // Stop VAD (also stops the mic stream)
        if (this.vad) {
            this.vad.pause();
            this.vad.destroy().catch(() => {});
            this.vad = null;
        }
        if (this.stream) {
            this.stream.getTracks().forEach(t => t.stop());
            this.stream = null;
        }

        this._processSpeech();
    },

    // ── Assemble speech frames → WAV → send to /extract-mfcc ─────────────
    async _processSpeech() {
        if (this.speechFrames.length === 0) {
            this._setStatus("❌ No speech detected. Try again.", "text-red-400");
            this._showStartBtn("🎤 Try Again");
            return;
        }

        // Concatenate all speech-only frames
        const totalSamples = this.speechFrames.reduce((s, f) => s + f.length, 0);
        const allSamples   = new Float32Array(totalSamples);
        let offset = 0;
        for (const frame of this.speechFrames) {
            allSamples.set(frame, offset);
            offset += frame.length;
        }

        const durationMs = (totalSamples / this.TARGET_SAMPLE_RATE) * 1000;
        console.log(`[SpeechCapture] Speech frames: ${totalSamples} samples @ 16kHz = ${(durationMs/1000).toFixed(2)}s`);

        if (durationMs < this.MIN_SPEECH_MS) {
            this._setStatus("❌ Too short — speak the full phrase and try again.", "text-red-400");
            this._showStartBtn("🎤 Try Again");
            return;
        }

        // VAD already outputs 16kHz — no resampling needed
        const finalSamples = this._autoGain(allSamples);
        const wavBlob      = this._float32ToWav(finalSamples, this.TARGET_SAMPLE_RATE);
        const base64Audio  = await this._blobToBase64(wavBlob);

        console.log(`[SpeechCapture] WAV: ${wavBlob.size} bytes  duration=${(durationMs/1000).toFixed(2)}s`);
        this._setDiag(`Captured ${(durationMs/1000).toFixed(1)}s of clean speech`);

        const username = typeof authUsername    !== "undefined" ? authUsername
                       : typeof currentUsername !== "undefined" ? currentUsername
                       : "";

        try {
            const response = await fetch(`${API_BASE}/enroll/extract-mfcc`, {
                method:  "POST",
                headers: { "Content-Type": "application/json" },
                body:    JSON.stringify({
                    audio_data:   base64Audio,
                    audio_format: "wav",
                    username,
                })
            });

            const result = await response.json();
            console.log("[SpeechCapture] extract-mfcc:", result);
            console.log("[SpeechCapture] ecapa_embedding length:", (result.ecapa_embedding || []).length);

            if (!result.success) {
                const detail = result.detail || "Processing failed";
                const snrTxt = result.snr_db != null ? ` (SNR: ${result.snr_db.toFixed(1)} dB)` : "";
                this._setStatus(`❌ ${detail}${snrTxt}`, "text-red-400");
                this._showStartBtn("🎤 Try Again");
                return;
            }

            // Quality feedback
            if (result.snr_db != null) {
                const q   = result.snr_db > 25 ? "excellent"
                          : result.snr_db > 15 ? "good"
                          : result.snr_db > 8  ? "acceptable" : "poor";
                this._setDiag(`Audio quality: ${q} — SNR=${result.snr_db.toFixed(1)}dB  voiced=${(result.voiced_fraction * 100).toFixed(0)}%`);
            }

            const fullFeatureDict = {
                mfcc_features:          result.mfcc_features          || [],
                mfcc_std:               result.mfcc_std               || [],
                delta_mfcc_mean:        result.delta_mfcc_mean        || [],
                delta2_mfcc_mean:       result.delta2_mfcc_mean       || [],
                pitch_mean:             result.pitch_mean             || 0,
                pitch_std:              result.pitch_std              || 0,
                speaking_rate:          result.speaking_rate          || 0,
                energy_mean:            result.energy_mean            || 0,
                energy_std:             result.energy_std             || 0,
                zcr_mean:               result.zcr_mean               || 0,
                spectral_centroid_mean: result.spectral_centroid_mean || 0,
                spectral_rolloff_mean:  result.spectral_rolloff_mean  || 0,
                spectral_flux_mean:     result.spectral_flux_mean     || 0,
                voiced_fraction:        result.voiced_fraction        || 0,
                snr_db:                 result.snr_db                 || 0,
                ecapa_embedding:        result.ecapa_embedding        || [],
                raw_audio_b64:          base64Audio,
            };

            if (typeof onVoiceAuthComplete === "function") {
                this._setStatus("⏳ Verifying voice…", "text-yellow-400");
                onVoiceAuthComplete(fullFeatureDict);

            } else if (typeof onVoiceRecorded === "function") {
                this._setStatus(`✅ Recording ${this.currentAttempt} captured!`, "text-green-400");
                onVoiceRecorded(fullFeatureDict);
                this.currentAttempt++;

                const btn = document.getElementById("record-btn");
                if (btn) {
                    if (this.currentAttempt <= this.maxAttempts) {
                        btn.disabled    = false;
                        btn.textContent = `🎤 Record ${this.currentAttempt}/${this.maxAttempts}`;
                        btn.onclick     = startRecording;
                    } else {
                        btn.disabled    = true;
                        btn.textContent = "✅ All recordings done";
                    }
                }
            }

        } catch (err) {
            console.error("[SpeechCapture] API error:", err);
            this._setStatus("❌ Could not connect to server.", "text-red-400");
            this._showStartBtn("🎤 Try Again");
        }
    },

    async reset() {
        if (this._watchdogTimer) {
            clearTimeout(this._watchdogTimer);
            this._watchdogTimer = null;
        }
        this.isRecording  = false;
        this._vadStarted  = false;
        this.speechFrames = [];
        if (this.vad) {
            this.vad.pause();
            this.vad.destroy().catch(() => {});
            this.vad = null;
        }
        if (this.stream) {
            this.stream.getTracks().forEach(t => t.stop());
            this.stream = null;
        }
        // Reset mic visual to idle state
        this._setIndicator(false, null);
    }
};


// ── Global startRecording() — called by record-btn onclick ───────────────────
async function startRecording() {
    const btn     = document.getElementById("record-btn");
    const diagTxt = document.getElementById("diag-text");

    if (btn) {
        btn.disabled    = true;
        btn.textContent = "🎤 Starting…";
    }
    if (diagTxt) diagTxt.textContent = "";

    await SpeechCapture.reset();
    await SpeechCapture.startRecording();
}