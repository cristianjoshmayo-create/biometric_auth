// frontend/js/speech.js
// v7 — Universal MediaRecorder + Noise Handling (cross-browser)
//
// Based on v6 (MediaRecorder, works on ALL browsers) with noise handling
// ported back from v5:
//   1. Auto-gain normalisation  — scales up quiet recordings before sending
//      to the backend so MFCC extraction gets consistent energy levels.
//   2. Noise floor pre-check   — measures ambient RMS for 500ms before
//      recording. Warns user if background noise is too high (> -25 dBFS).
//   3. Enhanced getUserMedia   — adds Chrome-specific noise suppression hints
//      (googNoiseSuppression2, googHighpassFilter) on top of the standard
//      noiseSuppression / echoCancellation flags already in v6.
//   4. Audio quality logging   — SNR / voiced-fraction feedback in diag bar.
//
// NOTE: RNNoise AudioWorklet was intentionally NOT ported back — it requires
//       SharedArrayBuffer (special HTTP headers) and breaks Firefox/Safari.
//       Server-side noisereduce on /extract-mfcc is the recommended substitute
//       for speech-babble noise (people talking in the background).
//
// Works on: Chrome, Edge, Opera, Firefox, Safari, all mobile browsers.
//
// Integration contract (unchanged from v4/v5/v6):
//   - Global startRecording() called by record-btn onclick
//   - Calls onVoiceRecorded(featureDict)     during enrollment
//   - Calls onVoiceAuthComplete(featureDict) during login
//   - DOM ids: record-btn, voice-status, diag-text, recording-indicator

const SpeechCapture = {
    mediaRecorder:   null,
    stream:          null,
    audioChunks:     [],
    isRecording:     false,
    currentAttempt:  1,
    maxAttempts:     3,
    _watchdogTimer:  null,

    MAX_DURATION_MS:  8000,   // auto-stop after 8 seconds
    MIN_DURATION_MS:  1500,   // reject recordings shorter than 1.5s
    NOISE_CHECK_MS:   500,    // ms to sample background noise before recording
    NOISE_THRESHOLD:  -25,    // dBFS — warn user if ambient noise exceeds this

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

    _setIndicator(active) {
        const el = document.getElementById("recording-indicator");
        if (!el) return;
        el.style.backgroundColor = active ? "#10b981" : "#f59e0b";
        if (active) el.classList.remove("animate-pulse");
        else        el.classList.add("animate-pulse");
    },

    // ── Pick the best supported MIME type ────────────────────────────────
    _getMimeType() {
        const types = [
            "audio/wav",
            "audio/webm;codecs=pcm",
            "audio/webm",
            "audio/ogg;codecs=opus",
            "audio/ogg",
            "audio/mp4",
        ];
        for (const t of types) {
            if (MediaRecorder.isTypeSupported(t)) return t;
        }
        return "";   // browser default
    },

    // ── Noise floor pre-check (ported from v5) ────────────────────────────
    // Opens the mic briefly, measures ambient RMS, warns if too noisy.
    // Returns { ok: bool, rms: float, dbfs: float }
    async _checkNoiseFloor(stream) {
        try {
            const ctx      = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: 16000 });
            const src      = ctx.createMediaStreamSource(stream);
            const analyser = ctx.createAnalyser();
            analyser.fftSize = 2048;
            src.connect(analyser);

            await new Promise(r => setTimeout(r, this.NOISE_CHECK_MS));

            const buf = new Float32Array(analyser.fftSize);
            analyser.getFloatTimeDomainData(buf);
            const rms  = Math.sqrt(buf.reduce((s, x) => s + x * x, 0) / buf.length);
            const dbfs = rms > 0 ? 20 * Math.log10(rms) : -100;
            await ctx.close();

            console.log(`[SpeechCapture] Noise floor: ${dbfs.toFixed(1)} dBFS (RMS=${rms.toFixed(4)})`);
            return { ok: dbfs < this.NOISE_THRESHOLD, rms, dbfs };
        } catch (e) {
            // AudioContext may be unavailable in some environments — non-fatal
            console.warn("[SpeechCapture] Noise floor check failed (non-fatal):", e.message);
            return { ok: true, rms: 0, dbfs: -100 };
        }
    },

    // ── Auto-gain normalisation (ported from v5) ──────────────────────────
    // Converts the recorded Blob to Float32, scales up if RMS < targetRms,
    // then re-encodes as WAV and returns a new Blob + base64 string.
    // Falls back to original Blob if AudioContext / decoding fails.
    async _autoGainBlob(blob) {
        try {
            const ctx        = new (window.AudioContext || window.webkitAudioContext)();
            const arrayBuf   = await blob.arrayBuffer();
            const audioBuf   = await ctx.decodeAudioData(arrayBuf);
            await ctx.close();

            // Get mono samples (mix down if multi-channel)
            const samples = audioBuf.getChannelData(0);
            const rms     = Math.sqrt(samples.reduce((s, x) => s + x * x, 0) / samples.length);

            if (rms < 0.001) {
                console.warn("[SpeechCapture] Audio appears silent (RMS=" + rms.toFixed(5) + ")");
                return { blob, base64: await this._blobToBase64(blob), gained: false };
            }

            const targetRms  = 0.08;
            const maxGain    = 10.0;
            const gain       = Math.min(targetRms / rms, maxGain);

            if (rms >= targetRms) {
                // Already loud enough — skip gain, just return base64 of original
                console.log(`[SpeechCapture] Auto-gain: not needed (RMS=${rms.toFixed(4)})`);
                return { blob, base64: await this._blobToBase64(blob), gained: false };
            }

            console.log(`[SpeechCapture] Auto-gain: ×${gain.toFixed(2)} (RMS ${rms.toFixed(4)} → ${(rms * gain).toFixed(4)})`);

            // Apply gain and clip
            const gained = new Float32Array(samples.length);
            for (let i = 0; i < samples.length; i++) {
                gained[i] = Math.max(-1, Math.min(1, samples[i] * gain));
            }

            // Re-encode as WAV
            const wavBlob = this._float32ToWav(gained, audioBuf.sampleRate);
            const base64  = await this._blobToBase64(wavBlob);
            return { blob: wavBlob, base64, gained: true };

        } catch (e) {
            console.warn("[SpeechCapture] Auto-gain failed (non-fatal):", e.message);
            return { blob, base64: await this._blobToBase64(blob), gained: false };
        }
    },

    // ── Float32 → 16-bit PCM WAV Blob (ported from v5) ───────────────────
    _float32ToWav(samples, sampleRate) {
        const numCh    = 1;
        const bps      = 16;
        const bpSamp   = bps / 8;
        const blkAlign = numCh * bpSamp;
        const byteRate = sampleRate * blkAlign;
        const dataSize = samples.length * bpSamp;
        const buf      = new ArrayBuffer(44 + dataSize);
        const v        = new DataView(buf);
        const ws       = (o, s) => { for (let i = 0; i < s.length; i++) v.setUint8(o + i, s.charCodeAt(i)); };

        ws(0,  "RIFF");  v.setUint32( 4, 36 + dataSize, true);
        ws(8,  "WAVE");  ws(12, "fmt ");
        v.setUint32(16, 16, true);
        v.setUint16(20,  1, true);
        v.setUint16(22, numCh, true);
        v.setUint32(24, sampleRate, true);
        v.setUint32(28, byteRate, true);
        v.setUint16(32, blkAlign, true);
        v.setUint16(34, bps, true);
        ws(36, "data");  v.setUint32(40, dataSize, true);

        let off = 44;
        for (let i = 0; i < samples.length; i++, off += 2) {
            const s = Math.max(-1, Math.min(1, samples[i]));
            v.setInt16(off, s < 0 ? s * 0x8000 : s * 0x7FFF, true);
        }
        return new Blob([buf], { type: "audio/wav" });
    },

    // ── Main entry point ──────────────────────────────────────────────────
    async startRecording() {
        try {
            this._setStatus("⏳ Requesting microphone…", "text-yellow-400");
            this._setDiag("");

            // Enhanced getUserMedia — standard flags + Chrome-specific hints.
            // Non-standard goog* keys are silently ignored by Firefox/Safari.
            this.stream = await navigator.mediaDevices.getUserMedia({
                audio: {
                    channelCount:              1,
                    sampleRate:                { ideal: 16000 },
                    echoCancellation:          true,
                    noiseSuppression:          true,   // WebRTC NS (all browsers)
                    autoGainControl:           false,  // we handle gain ourselves
                    // Chrome / Edge enhanced noise suppression hints
                    googNoiseSuppression:      true,
                    googNoiseSuppression2:     true,
                    googHighpassFilter:        true,
                    googAudioMirroring:        false,
                }
            });

            // ── Noise floor check ─────────────────────────────────────────
            this._setStatus("🔍 Checking background noise…", "text-yellow-400");
            const { ok: noiseOk, dbfs } = await this._checkNoiseFloor(this.stream);

            if (!noiseOk) {
                // Warn but don't block — server-side noisereduce will still help
                this._setStatus(
                    `⚠️ Background noise detected (${dbfs.toFixed(0)} dBFS). ` +
                    "Try to move to a quieter spot, then speak clearly.",
                    "text-yellow-400"
                );
                this._setDiag(`Ambient noise: ${dbfs.toFixed(1)} dBFS (ideal < ${this.NOISE_THRESHOLD} dBFS) — proceeding anyway`);
                console.warn("[SpeechCapture] High ambient noise, proceeding anyway");
                // Short pause so user reads the warning before recording starts
                await new Promise(r => setTimeout(r, 1500));
            }

            // ── MediaRecorder setup ───────────────────────────────────────
            const mimeType = this._getMimeType();
            const options  = mimeType ? { mimeType } : {};

            this.audioChunks   = [];
            this.mediaRecorder = new MediaRecorder(this.stream, options);

            this.mediaRecorder.ondataavailable = (e) => {
                if (e.data && e.data.size > 0) {
                    this.audioChunks.push(e.data);
                }
            };

            this.mediaRecorder.onstop = async () => {
                await this._processChunks(mimeType);
            };

            // Auto-stop after MAX_DURATION_MS
            this._watchdogTimer = setTimeout(() => {
                if (this.isRecording) {
                    console.log("[SpeechCapture] Auto-stop after max duration");
                    this.stopRecording();
                }
            }, this.MAX_DURATION_MS);

            this.mediaRecorder.start(100);   // collect chunks every 100ms
            this.isRecording = true;

            this._setStatus("🔴 Recording… click Stop when done speaking", "text-red-400");
            this._setDiag("Speak your phrase clearly into the microphone");
            this._setIndicator(true);

            // Update button to show Stop
            const btn = document.getElementById("record-btn");
            if (btn) {
                btn.textContent = "⏹ Stop Recording";
                btn.disabled    = false;
                btn.onclick     = () => this.stopRecording();
                btn.classList.replace("bg-yellow-600", "bg-red-600");
            }

            return true;

        } catch (err) {
            console.error("[SpeechCapture] Init error:", err);
            const msg = err.name === "NotAllowedError"
                ? "❌ Microphone access denied. Please allow microphone and try again."
                : `❌ Microphone error: ${err.message}`;
            this._setStatus(msg, "text-red-400");
            return false;
        }
    },

    // ── Stop recording (called by Stop button or watchdog) ────────────────
    stopRecording() {
        if (this._watchdogTimer) {
            clearTimeout(this._watchdogTimer);
            this._watchdogTimer = null;
        }
        if (this.mediaRecorder && this.isRecording) {
            this.isRecording = false;
            this._setStatus("⚙️ Processing audio…", "text-yellow-400");
            this._setDiag("");
            this._setIndicator(false);
            this.mediaRecorder.stop();
        }
    },

    // ── Process recorded chunks → auto-gain → send to /extract-mfcc ──────
    async _processChunks(mimeType) {
        if (this.stream) {
            this.stream.getTracks().forEach(t => t.stop());
            this.stream = null;
        }

        if (this.audioChunks.length === 0) {
            this._setStatus("❌ No audio recorded. Try again.", "text-red-400");
            this._resetBtn();
            return;
        }

        const actualMime = mimeType || "audio/webm";
        const rawBlob    = new Blob(this.audioChunks, { type: actualMime });

        // Reject obviously too-short recordings by size
        if (rawBlob.size < 8000) {
            this._setStatus(
                "❌ Recording too short. Hold the button and speak the full phrase.",
                "text-red-400"
            );
            this._resetBtn();
            return;
        }

        console.log(`[SpeechCapture] Recorded blob: ${rawBlob.size} bytes  type=${actualMime}`);

        // ── Auto-gain normalisation ───────────────────────────────────────
        this._setDiag("Normalising audio levels…");
        const { blob, base64: base64Audio, gained } = await this._autoGainBlob(rawBlob);
        if (gained) {
            this._setDiag("Auto-gain applied — audio normalised");
        }

        // Determine format string for server
        // If auto-gain ran, blob is now WAV regardless of original mimeType
        let audioFormat = gained ? "wav" : "webm";
        if (!gained) {
            if (actualMime.includes("wav"))       audioFormat = "wav";
            else if (actualMime.includes("ogg"))  audioFormat = "ogg";
            else if (actualMime.includes("mp4"))  audioFormat = "mp4";
        }

        const username = typeof authUsername    !== "undefined" ? authUsername
                       : typeof currentUsername !== "undefined" ? currentUsername
                       : "";

        try {
            const response = await fetch(`${API_BASE}/enroll/extract-mfcc`, {
                method:  "POST",
                headers: { "Content-Type": "application/json" },
                body:    JSON.stringify({
                    audio_data:   base64Audio,
                    audio_format: audioFormat,
                    username,
                })
            });

            const result = await response.json();
            console.log("[SpeechCapture] extract-mfcc result:", result);
            console.log("[SpeechCapture] ecapa_embedding length:", (result.ecapa_embedding || []).length);

            if (!result.success) {
                const detail = result.detail || "Processing failed";
                const snrTxt = result.snr_db != null
                    ? ` (SNR: ${result.snr_db.toFixed(1)} dB)`
                    : "";
                this._setStatus(`❌ ${detail}${snrTxt}`, "text-red-400");
                this._resetBtn();
                return;
            }

            // ── Audio quality feedback (ported from v5) ───────────────────
            if (result.snr_db != null) {
                const q   = result.snr_db > 25 ? "excellent"
                          : result.snr_db > 15 ? "good"
                          : result.snr_db > 8  ? "acceptable" : "poor";
                const msg = `Audio quality: ${q} — SNR=${result.snr_db.toFixed(1)}dB  voiced=${(result.voiced_fraction * 100).toFixed(0)}%`;
                console.log(`[SpeechCapture] ${msg}`);
                this._setDiag(msg);
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
                this._setStatus(`✅ Recording ${this.currentAttempt} saved!`, "text-green-400");
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
            console.error("[SpeechCapture] API error:", err);
            this._setStatus("❌ Could not connect to server.", "text-red-400");
            this._resetBtn();
        }
    },

    _blobToBase64(blob) {
        return new Promise((resolve, reject) => {
            const r     = new FileReader();
            r.onloadend = () => resolve(r.result.split(",")[1]);
            r.onerror   = reject;
            r.readAsDataURL(blob);
        });
    },

    _resetBtn() {
        const btn = document.getElementById("record-btn");
        if (btn) {
            btn.disabled    = false;
            btn.textContent = "🎤 Try Again";
            btn.onclick     = startRecording;
            if (!btn.classList.contains("bg-red-600"))
                btn.classList.replace("bg-gray-600", "bg-red-600");
        }
    },

    async reset() {
        if (this._watchdogTimer) {
            clearTimeout(this._watchdogTimer);
            this._watchdogTimer = null;
        }
        if (this.mediaRecorder && this.isRecording) {
            this.mediaRecorder.stop();
        }
        if (this.stream) {
            this.stream.getTracks().forEach(t => t.stop());
            this.stream = null;
        }
        this.mediaRecorder = null;
        this.audioChunks   = [];
        this.isRecording   = false;
    }
};


// ── Global startRecording() — called by record-btn onclick ───────────────────
async function startRecording() {
    const btn       = document.getElementById("record-btn");
    const indicator = document.getElementById("recording-indicator");
    const diagTxt   = document.getElementById("diag-text");

    if (btn) {
        btn.disabled    = true;
        btn.textContent = "🎤 Initialising…";
        btn.classList.replace("bg-red-600",  "bg-yellow-600");
        btn.classList.replace("bg-gray-600", "bg-yellow-600");
    }
    if (indicator) {
        indicator.classList.remove("hidden");
        indicator.style.backgroundColor = "#f59e0b";
        indicator.classList.add("animate-pulse");
    }
    if (diagTxt) diagTxt.textContent = "";

    await SpeechCapture.reset();

    await SpeechCapture.startRecording();
}