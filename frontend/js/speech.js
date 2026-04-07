// frontend/js/speech.js
// v12 — Zero-dependency VAD using Web Audio API only
//
// Removes @ricky0123/vad-web and onnxruntime-web entirely.
// Uses ScriptProcessorNode energy detection to find speech boundaries:
//   - RMS energy above threshold  → speech frame
//   - 400ms of silence after speech → auto-stop
//   - 10s watchdog hard stop
//
// Works on: Chrome, Firefox, Safari, Edge, Opera, iOS Safari, Android Chrome.
// No WASM, no ES modules, no CDN dependencies.
//
// Integration contract (unchanged from v4-v9):
//   - Global startRecording() called by record-btn onclick
//   - Calls onVoiceRecorded(featureDict)     during enrollment
//   - Calls onVoiceAuthComplete(featureDict) during login

const SpeechCapture = {
    audioCtx:       null,
    stream:         null,
    processor:      null,
    analyser:       null,
    allSamples:     [],
    isRecording:    false,
    _hasSpeech:     false,
    _silenceMs:     0,
    _watchdogTimer: null,
    _nativeSr:      44100,
    currentAttempt: 1,
    maxAttempts:    3,

    TARGET_SR:       16000,
    FRAME_SIZE:      4096,
    SPEECH_THRESH:   0.01,
    SILENCE_MS:      400,
    MIN_SPEECH_MS:   800,
    MAX_DURATION_MS: 10000,

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
    _setIndicator(state) {
        const circle = document.getElementById("mic-circle");
        const svg    = document.getElementById("mic-svg");
        const ring1  = document.getElementById("mic-ring-1");
        const ring2  = document.getElementById("mic-ring-2");
        if (!circle) return;
        if (state === "speaking") {
            circle.style.background = "#10b981";
            if (svg)   svg.setAttribute("stroke", "#ffffff");
            if (ring1) { ring1.style.opacity = "0.6"; ring1.style.transform = "scale(1)"; }
            if (ring2) { ring2.style.opacity = "0.3"; ring2.style.transform = "scale(1)"; }
        } else if (state === "waiting") {
            circle.style.background = "#374151";
            if (svg)   svg.setAttribute("stroke", "#f59e0b");
            if (ring1) { ring1.style.opacity = "0"; ring1.style.transform = "scale(0.75)"; }
            if (ring2) { ring2.style.opacity = "0"; ring2.style.transform = "scale(0.75)"; }
        } else {
            circle.style.background = "#374151";
            if (svg)   svg.setAttribute("stroke", "#9ca3af");
            if (ring1) { ring1.style.opacity = "0"; ring1.style.transform = "scale(0.75)"; }
            if (ring2) { ring2.style.opacity = "0"; ring2.style.transform = "scale(0.75)"; }
        }
    },
    _showStopBtn() {
        const circle = document.getElementById("mic-circle");
        const svg    = document.getElementById("mic-svg");
        if (circle) { circle.style.background = "#dc2626"; circle.onclick = () => this.stopRecording(); }
        if (svg)    svg.setAttribute("stroke", "#ffffff");
        const btn = document.getElementById("record-btn");
        if (btn)  { btn.disabled = false; btn.textContent = "⏹ Stop Recording"; btn.onclick = () => this.stopRecording(); }
        const tryAgain = document.getElementById("try-again-btn");
        if (tryAgain) tryAgain.disabled = true;
    },
    _showStartBtn(label) {
        const circle = document.getElementById("mic-circle");
        const svg    = document.getElementById("mic-svg");
        if (circle) { circle.style.background = "#374151"; circle.onclick = startRecording; }
        if (svg)    svg.setAttribute("stroke", "#9ca3af");
        const btn = document.getElementById("record-btn");
        if (btn)  { btn.disabled = false; btn.textContent = label || "🎤 Start Recording"; btn.onclick = startRecording; }
        const tryAgain = document.getElementById("try-again-btn");
        if (tryAgain) tryAgain.disabled = false;
    },

    _rms(buf) {
        let sum = 0;
        for (let i = 0; i < buf.length; i++) sum += buf[i] * buf[i];
        return Math.sqrt(sum / buf.length);
    },

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

    _autoGain(samples) {
        const rms = Math.sqrt(samples.reduce((s, x) => s + x * x, 0) / samples.length);
        if (rms < 0.001 || rms >= 0.08) return samples;
        const gain = Math.min(0.08 / rms, 10.0);
        console.log(`[SpeechCapture] Auto-gain x${gain.toFixed(2)}`);
        const out = new Float32Array(samples.length);
        for (let i = 0; i < samples.length; i++) out[i] = Math.max(-1, Math.min(1, samples[i] * gain));
        return out;
    },

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
            const r = new FileReader();
            r.onloadend = () => resolve(r.result.split(",")[1]);
            r.onerror   = reject;
            r.readAsDataURL(blob);
        });
    },

    async startRecording() {
        try {
            this._setStatus("⏳ Requesting microphone…", "text-yellow-400");
            this._setDiag("");
            this.allSamples = [];
            this._hasSpeech = false;
            this._silenceMs = 0;

            this.stream = await navigator.mediaDevices.getUserMedia({
                audio: { channelCount: 1, echoCancellation: false, noiseSuppression: false, autoGainControl: false }
            });

            this.audioCtx  = new (window.AudioContext || window.webkitAudioContext)();
            this._nativeSr = this.audioCtx.sampleRate;
            const source   = this.audioCtx.createMediaStreamSource(this.stream);
            this.analyser  = this.audioCtx.createAnalyser();
            this.analyser.fftSize = 256;
            this.processor = this.audioCtx.createScriptProcessor(this.FRAME_SIZE, 1, 1);

            source.connect(this.analyser);
            this.analyser.connect(this.processor);
            this.processor.connect(this.audioCtx.destination);

            this.processor.onaudioprocess = (e) => {
                if (!this.isRecording) return;
                const input = e.inputBuffer.getChannelData(0);
                this.allSamples.push(new Float32Array(input));
                const energy   = this._rms(input);
                const isSpeech = energy > this.SPEECH_THRESH;
                // Yellow while waiting for voice, green once voice is detected
                if (isSpeech && !this._hasSpeech) {
                    this._hasSpeech = true;
                    this._setIndicator("speaking");  // turn green
                    this._setStatus("🔴 Recording… click Stop when done", "text-green-400");
                }
                // Pulse rings with volume
                const ring1 = document.getElementById("mic-ring-1");
                const ring2 = document.getElementById("mic-ring-2");
                if (ring1) ring1.style.opacity = Math.min(energy * 20, 0.8).toFixed(2);
                if (ring2) ring2.style.opacity = Math.min(energy * 10, 0.4).toFixed(2);
            };

            this.isRecording = true;
            this._watchdogTimer = setTimeout(() => {
                if (this.isRecording) { console.log("[SpeechCapture] Watchdog"); this.stopRecording(); }
            }, this.MAX_DURATION_MS);

            this._setStatus("🎤 Listening… speak your phrase", "text-blue-400");
            this._setIndicator("waiting");
            this._setDiag("Speak your phrase, then press Stop Recording");
            this._showStopBtn();
            return true;

        } catch (err) {
            console.error("[SpeechCapture] Init error:", err);
            const msg = err.name === "NotAllowedError"
                ? "❌ Microphone access denied. Allow microphone and try again."
                : `❌ Could not start recording: ${err.message}`;
            this._setStatus(msg, "text-red-400");
            this._showStartBtn("🎤 Try Again");
            return false;
        }
    },

    stopRecording() {
        if (this._watchdogTimer) { clearTimeout(this._watchdogTimer); this._watchdogTimer = null; }
        if (!this.isRecording) return;
        this.isRecording = false;
        this._setStatus("⚙️ Processing captured voice…", "text-yellow-400");
        this._setDiag("");
        this._setIndicator("idle");
        if (this.processor) { this.processor.disconnect(); this.processor = null; }
        if (this.analyser)  { this.analyser.disconnect();  this.analyser  = null; }
        if (this.audioCtx)  { this.audioCtx.close();       this.audioCtx  = null; }
        if (this.stream)    { this.stream.getTracks().forEach(t => t.stop()); this.stream = null; }
        this._processSpeech();
    },

    async _processSpeech() {
        if (this.allSamples.length === 0) {
            this._setStatus("❌ No audio captured. Try again.", "text-red-400");
            this._showStartBtn("🎤 Try Again");
            return;
        }

        const totalSamples = this.allSamples.reduce((s, f) => s + f.length, 0);
        const raw = new Float32Array(totalSamples);
        let offset = 0;
        for (const chunk of this.allSamples) { raw.set(chunk, offset); offset += chunk.length; }

        // Trim leading/trailing silence with 200ms padding
        const padFrames = Math.round(this._nativeSr * 0.2);
        let start = 0, end = raw.length - 1;
        for (let i = 0; i < raw.length; i++)         { if (Math.abs(raw[i]) > this.SPEECH_THRESH * 0.5) { start = Math.max(0, i - padFrames); break; } }
        for (let i = raw.length - 1; i >= 0; i--)    { if (Math.abs(raw[i]) > this.SPEECH_THRESH * 0.5) { end = Math.min(raw.length - 1, i + padFrames); break; } }

        const trimmed    = raw.slice(start, end + 1);
        const resampled  = this._resample(trimmed, this._nativeSr, this.TARGET_SR);
        const durationMs = (resampled.length / this.TARGET_SR) * 1000;
        console.log(`[SpeechCapture] ${(durationMs/1000).toFixed(2)}s after trim+resample`);

        if (durationMs < this.MIN_SPEECH_MS) {
            this._setStatus("❌ Too short — speak the full phrase and try again.", "text-red-400");
            this._showStartBtn("🎤 Try Again");
            return;
        }

        const finalSamples = this._autoGain(resampled);
        const wavBlob      = this._float32ToWav(finalSamples, this.TARGET_SR);
        const base64Audio  = await this._blobToBase64(wavBlob);
        this._setDiag(`Captured ${(durationMs/1000).toFixed(1)}s of speech`);

        const username = typeof authUsername    !== "undefined" ? authUsername
                       : typeof currentUsername !== "undefined" ? currentUsername
                       : "";
        try {
            const response = await fetch(`${API_BASE}/enroll/extract-mfcc`, {
                method:  "POST",
                headers: { "Content-Type": "application/json" },
                body:    JSON.stringify({ audio_data: base64Audio, audio_format: "wav", username })
            });
            const result = await response.json();
            console.log("[SpeechCapture] extract-mfcc:", result);

            if (!result.success) {
                const snrTxt = result.snr_db != null ? ` (SNR: ${result.snr_db.toFixed(1)} dB)` : "";
                this._setStatus(`❌ ${result.detail || "Processing failed"}${snrTxt}`, "text-red-400");
                this._showStartBtn("🎤 Try Again");
                return;
            }

            if (result.snr_db != null) {
                const q = result.snr_db > 25 ? "excellent" : result.snr_db > 15 ? "good" : result.snr_db > 8 ? "acceptable" : "poor";
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
        if (this._watchdogTimer) { clearTimeout(this._watchdogTimer); this._watchdogTimer = null; }
        this.isRecording = false;
        this._hasSpeech  = false;
        this._silenceMs  = 0;
        this.allSamples  = [];
        if (this.processor) { this.processor.disconnect(); this.processor = null; }
        if (this.analyser)  { this.analyser.disconnect();  this.analyser  = null; }
        if (this.audioCtx)  { this.audioCtx.close().catch(() => {}); this.audioCtx = null; }
        if (this.stream)    { this.stream.getTracks().forEach(t => t.stop()); this.stream = null; }
        this._setIndicator("idle");
        this._showStartBtn("🎤 Start Recording");   // ← ADD THIS
        const status = document.getElementById("voice-status");
        if (status) status.textContent = "Press the mic button and say your phrase.";  // ← ADD THIS
    }
};

async function startRecording() {
    const btn     = document.getElementById("record-btn");
    const diagTxt = document.getElementById("diag-text");
    if (btn)     { btn.disabled = true; btn.textContent = "🎤 Starting…"; }
    if (diagTxt) diagTxt.textContent = "";
    await SpeechCapture.reset();
    await SpeechCapture.startRecording();
}