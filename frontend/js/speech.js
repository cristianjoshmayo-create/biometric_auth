// frontend/js/speech.js
// v4 — Silero VAD replacement
//
// Root cause of v3 enrollment failures:
//   Hand-tuned RMS threshold VAD is fragile across mic hardware, browsers,
//   room noise, and speaking distance. WebRTC VAD (used server-side) only
//   catches ~50% of real speech frames at a 5% false-positive rate.
//
// Fix: replace custom VAD with @ricky0123/vad-web, which runs Silero VAD
//   via ONNX Runtime Web entirely in the browser. Silero VAD achieves ~88%
//   true-positive rate under the same conditions — 4x fewer missed frames.
//
// Integration contract (unchanged from v3):
//   - Calls onVoiceRecorded(featureDict)  during enrollment
//   - Calls onVoiceAuthComplete(featureDict) during login
//   - DOM ids used: record-btn, voice-status, diag-text, recording-indicator
//   - Global startRecording() function preserved
//   - Sends audio to POST /enroll/extract-mfcc, same as before

// ── CDN deps (loaded in HTML before this file) ───────────────────────────
// <script src="https://cdn.jsdelivr.net/npm/onnxruntime-web@1.22.0/dist/ort.wasm.min.js"></script>
// <script src="https://cdn.jsdelivr.net/npm/@ricky0123/vad-web@0.0.29/dist/bundle.min.js"></script>

const SpeechCapture = {
    myvad:          null,
    isRecording:    false,
    currentAttempt: 1,
    maxAttempts:    3,

    // ── Silero VAD config ─────────────────────────────────────────────────
    // positiveSpeechThreshold: probability above which a frame is "speech"
    // negativeSpeechThreshold: probability below which a frame ends speech
    // minSpeechFrames: minimum consecutive speech frames before firing onSpeechEnd
    // preSpeechPadFrames: frames of audio before speech start to include
    VAD_CONFIG: {
        positiveSpeechThreshold: 0.50,
        negativeSpeechThreshold: 0.35,
        minSpeechFrames:         8,     // ~0.5s at 16kHz / 1536 frame size
        preSpeechPadFrames:      3,
        redemptionFrames:        10,
        onnxWASMBasePath:  "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.22.0/dist/",
        baseAssetPath:     "https://cdn.jsdelivr.net/npm/@ricky0123/vad-web@0.0.29/dist/",
    },

    // ── Update live status display ────────────────────────────────────────
    _setStatus(text, color) {
        const el = document.getElementById("voice-status");
        if (!el) return;
        el.textContent  = text;
        el.className    = `text-center text-sm mb-4 ${color}`;
    },

    _setDiag(text) {
        const el = document.getElementById("diag-text");
        if (el) el.textContent = text;
    },

    _setIndicator(active) {
        const el = document.getElementById("recording-indicator");
        if (!el) return;
        el.style.backgroundColor = active ? "#10b981" : "#f59e0b";
        if (active) el.classList.remove("animate-pulse");
        else        el.classList.add("animate-pulse");
    },

    // ── Main entry point ──────────────────────────────────────────────────
    async startRecording() {
        const status = document.getElementById("voice-status");

        // Guard: check Silero VAD library loaded
        if (typeof vad === "undefined" || !vad.MicVAD) {
            this._setStatus(
                "❌ Voice library failed to load. Check your internet connection.",
                "text-red-400"
            );
            return false;
        }

        try {
            this._setStatus("⏳ Initialising voice detector…", "text-yellow-400");
            this._setDiag("");

            this.myvad = await vad.MicVAD.new({
                ...this.VAD_CONFIG,

                onSpeechStart: () => {
                    console.log("[SileroVAD] Speech start");
                    this.isRecording = true;
                    this._setStatus(
                        "🔴 Recording — speak the full phrase clearly…",
                        "text-red-400"
                    );
                    this._setIndicator(true);
                    this._setDiag("🟢 Voice detected");
                },

                onSpeechEnd: async (audioFloat32) => {
                    console.log(`[SileroVAD] Speech end — ${audioFloat32.length} samples (${(audioFloat32.length/16000).toFixed(2)}s)`);
                    this._setStatus("⚙️ Processing audio…", "text-yellow-400");
                    this._setDiag("");
                    this._setIndicator(false);
                    this.isRecording = false;

                    // Stop VAD so mic is released
                    this.myvad.pause();

                    await this._processFloat32(audioFloat32);
                },

                onVADMisfire: () => {
                    console.log("[SileroVAD] Misfire — too short, keep speaking");
                    this._setStatus(
                        "⚠️ Too short — speak the full phrase without pausing",
                        "text-yellow-400"
                    );
                    this._setDiag("🔴 No voice — keep speaking");
                    this._setIndicator(false);
                    this.isRecording = false;
                },
            });

            this.myvad.start();
            this._setStatus(
                "🎤 Listening… speak when ready",
                "text-green-400"
            );
            this._setDiag("Waiting for speech…");
            return true;

        } catch (err) {
            console.error("[SileroVAD] Init error:", err);
            this._setStatus(`❌ Microphone error: ${err.message}`, "text-red-400");
            return false;
        }
    },

    // ── Convert Float32Array → WAV → base64 → send to /extract-mfcc ──────
    async _processFloat32(audioFloat32) {
        // Silero always outputs 16kHz mono Float32 — convert to 16-bit PCM WAV
        const wavBlob = this._float32ToWav(audioFloat32, 16000);

        // Minimum duration guard (backend requires ≥1s of speech)
        const durationSec = audioFloat32.length / 16000;
        if (durationSec < 1.0 || wavBlob.size < 3000) {
            this._setStatus(
                `❌ Recording too short (${durationSec.toFixed(1)}s). Speak the full phrase.`,
                "text-red-400"
            );
            this._resetBtn();
            return;
        }

        console.log(`[SileroVAD] WAV size: ${wavBlob.size} bytes  duration: ${durationSec.toFixed(2)}s`);

        const base64Audio = await this._blobToBase64(wavBlob);
        const username    = typeof authUsername    !== "undefined" ? authUsername
                          : typeof currentUsername !== "undefined" ? currentUsername
                          : "";

        try {
            const response = await fetch(`${API_BASE}/enroll/extract-mfcc`, {
                method:  "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    audio_data:   base64Audio,
                    audio_format: "wav",   // always WAV — no format ambiguity
                    username
                })
            });

            const result = await response.json();
            console.log("[SileroVAD] extract-mfcc result:", result);

            if (!result.success) {
                const detail = result.detail || "Processing failed";
                const snrTxt = result.snr_db != null
                    ? ` (SNR: ${result.snr_db.toFixed(1)} dB)`
                    : "";
                this._setStatus(`❌ ${detail}${snrTxt}`, "text-red-400");
                this._resetBtn();
                return;
            }

            // Log quality
            if (result.snr_db != null) {
                const q = result.snr_db > 25 ? "excellent"
                        : result.snr_db > 15 ? "good"
                        : result.snr_db > 8  ? "acceptable" : "poor";
                console.log(`[SileroVAD] Audio quality: ${q} (SNR=${result.snr_db.toFixed(1)}dB  voiced=${(result.voiced_fraction*100).toFixed(0)}%)`);
            }

            // Build full 62-feature dict (same as v3)
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
            };

            // Route to enrollment or authentication callback (same as v3)
            if (typeof onVoiceAuthComplete === "function") {
                this._setStatus("⏳ Verifying voice…", "text-yellow-400");
                onVoiceAuthComplete(fullFeatureDict);

            } else if (typeof onVoiceRecorded === "function") {
                this._setStatus(
                    `✅ Recording ${this.currentAttempt} saved!`,
                    "text-green-400"
                );
                onVoiceRecorded(fullFeatureDict);
                this.currentAttempt++;

                const btn = document.getElementById("record-btn");
                if (btn) {
                    if (this.currentAttempt <= this.maxAttempts) {
                        btn.textContent = `🎤 Record ${this.currentAttempt}/${this.maxAttempts}`;
                        btn.disabled    = false;
                        btn.onclick     = startRecording;
                    } else {
                        btn.disabled    = true;
                        btn.textContent = "✅ All recordings done";
                        btn.classList.replace("bg-red-600", "bg-gray-600");
                    }
                }
            }

        } catch (err) {
            console.error("[SileroVAD] API error:", err);
            this._setStatus("❌ Could not connect to server.", "text-red-400");
            this._resetBtn();
        }
    },

    // ── Float32Array → 16-bit PCM WAV Blob ───────────────────────────────
    // Silero outputs Float32 at 16kHz. Backend expects WAV.
    // This avoids the webm/ogg/mp4 format ambiguity that caused pydub errors.
    _float32ToWav(samples, sampleRate) {
        const numChannels = 1;
        const bitsPerSample = 16;
        const bytesPerSample = bitsPerSample / 8;
        const blockAlign = numChannels * bytesPerSample;
        const byteRate = sampleRate * blockAlign;
        const dataSize = samples.length * bytesPerSample;
        const bufferSize = 44 + dataSize;

        const buffer = new ArrayBuffer(bufferSize);
        const view   = new DataView(buffer);

        // WAV header
        const writeStr = (offset, str) => {
            for (let i = 0; i < str.length; i++)
                view.setUint8(offset + i, str.charCodeAt(i));
        };
        writeStr(0,  "RIFF");
        view.setUint32(4,  36 + dataSize, true);
        writeStr(8,  "WAVE");
        writeStr(12, "fmt ");
        view.setUint32(16, 16, true);            // chunk size
        view.setUint16(20, 1,  true);            // PCM
        view.setUint16(22, numChannels, true);
        view.setUint32(24, sampleRate, true);
        view.setUint32(28, byteRate, true);
        view.setUint16(32, blockAlign, true);
        view.setUint16(34, bitsPerSample, true);
        writeStr(36, "data");
        view.setUint32(40, dataSize, true);

        // PCM samples: clamp Float32 [-1, 1] → Int16
        let offset = 44;
        for (let i = 0; i < samples.length; i++, offset += 2) {
            const s = Math.max(-1, Math.min(1, samples[i]));
            view.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7FFF, true);
        }

        return new Blob([buffer], { type: "audio/wav" });
    },

    _blobToBase64(blob) {
        return new Promise((resolve, reject) => {
            const reader     = new FileReader();
            reader.onloadend = () => resolve(reader.result.split(",")[1]);
            reader.onerror   = reject;
            reader.readAsDataURL(blob);
        });
    },

    _resetBtn() {
        const btn = document.getElementById("record-btn");
        if (btn) {
            btn.disabled    = false;
            btn.textContent = "🎤 Try Again";
            if (!btn.classList.contains("bg-red-600"))
                btn.classList.replace("bg-gray-600", "bg-red-600");
        }
    },

    reset() {
        if (this.myvad) {
            try { this.myvad.destroy(); } catch (_) {}
            this.myvad = null;
        }
        this.isRecording    = false;
        this.currentAttempt = 1;
    }
};


// ── Global startRecording() — called by record-btn onclick ───────────────
function startRecording() {
    const btn       = document.getElementById("record-btn");
    const indicator = document.getElementById("recording-indicator");
    const diagTxt   = document.getElementById("diag-text");

    if (btn) {
        btn.disabled    = true;
        btn.textContent = "🎤 Initialising…";
        btn.classList.replace("bg-red-600", "bg-yellow-600");
    }
    if (indicator) {
        indicator.classList.remove("hidden");
        indicator.style.backgroundColor = "#f59e0b";
    }
    if (diagTxt) diagTxt.textContent = "";

    SpeechCapture.reset();
    SpeechCapture.startRecording().then(ok => {
        if (btn) {
            if (ok) {
                btn.textContent = "🎤 Listening…";
                btn.classList.replace("bg-yellow-600", "bg-red-600");
            } else {
                btn.disabled    = false;
                btn.textContent = "🎤 Try Again";
                btn.classList.replace("bg-yellow-600", "bg-red-600");
            }
        }
    });
}