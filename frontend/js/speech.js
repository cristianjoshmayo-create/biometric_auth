// frontend/js/speech.js
// v13 — AudioWorklet replaces deprecated ScriptProcessorNode
//
// ScriptProcessorNode (createScriptProcessor) is deprecated and broken in
// Opera GX — onaudioprocess never fires, so allSamples stays empty and no
// audio ever reaches the server, meaning no ECAPA profile is created.
//
// AudioWorklet is the modern W3C standard and works reliably on:
//   Chrome 66+, Firefox 76+, Edge 79+, Opera 53+, Opera GX, Safari 14.1+
//
// The rest of the integration contract is unchanged:
//   - Global startRecording() called by record-btn onclick
//   - Calls onVoiceRecorded(featureDict)     during enrollment
//   - Calls onVoiceAuthComplete(featureDict) during login

const SpeechCapture = {
    audioCtx:       null,
    stream:         null,
    workletNode:    null,   // replaces processor + analyser
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
        // Target RMS 0.08 — enough for backend SNR check (≥0.005) with headroom.
        // Skip if already loud enough or if signal is essentially silent (mic error).
        if (rms < 0.0005 || rms >= 0.15) return samples;
        const gain = Math.min(0.08 / rms, 8.0);
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

    // ── Build the AudioWorklet module as a Blob URL so no separate file is needed.
    // The worklet runs in its own thread (AudioWorkletGlobalScope), collecting
    // 128-sample chunks and posting them back to the main thread via MessagePort.
    // This is the modern replacement for the deprecated ScriptProcessorNode.
    _buildWorkletUrl() {
        const code = `
            class PCMCollector extends AudioWorkletProcessor {
                constructor() {
                    super();
                    // Buffer up to FRAME_SIZE samples before posting to avoid
                    // flooding the main thread with 128-sample messages.
                    this._buf  = new Float32Array(4096);
                    this._fill = 0;
                }
                process(inputs) {
                    const ch = inputs[0] && inputs[0][0];
                    if (!ch) return true;
                    for (let i = 0; i < ch.length; i++) {
                        this._buf[this._fill++] = ch[i];
                        if (this._fill === this._buf.length) {
                            // Post a copy — the original buffer is reused next frame
                            this.port.postMessage(this._buf.slice(0));
                            this._fill = 0;
                        }
                    }
                    return true;  // keep processor alive
                }
            }
            registerProcessor('pcm-collector', PCMCollector);
        `;
        const blob = new Blob([code], { type: "application/javascript" });
        return URL.createObjectURL(blob);
    },

    async startRecording() {
        try {
            this._setStatus("⏳ Requesting microphone…", "text-yellow-400");
            this._setDiag("");
            this.allSamples = [];
            this._hasSpeech = false;
            this._silenceMs = 0;

            this.stream = await navigator.mediaDevices.getUserMedia({
                audio: {
                    channelCount:      1,
                    echoCancellation:  true,   // remove mic echo from room reflections
                    noiseSuppression:  true,   // browser-level denoising (free, hardware-accelerated)
                    autoGainControl:   true,   // normalize mic volume across devices
                    sampleRate:        { ideal: 16000 },
                    sampleSize:        16,
                }
            });

            this.audioCtx  = new (window.AudioContext || window.webkitAudioContext)();
            this._nativeSr = this.audioCtx.sampleRate;

            // ── AudioWorklet setup (replaces createScriptProcessor) ──────────
            // Works on Chrome, Firefox, Edge, Opera GX, Safari 14.1+.
            // ScriptProcessorNode is deprecated and broken in Opera GX.
            const workletUrl = this._buildWorkletUrl();
            await this.audioCtx.audioWorklet.addModule(workletUrl);
            URL.revokeObjectURL(workletUrl);  // free the Blob URL immediately

            const source = this.audioCtx.createMediaStreamSource(this.stream);
            this.workletNode = new AudioWorkletNode(this.audioCtx, "pcm-collector");

            // Receive buffered PCM chunks from the worklet thread
            this.workletNode.port.onmessage = (e) => {
                if (!this.isRecording) return;
                const chunk  = new Float32Array(e.data);
                this.allSamples.push(chunk);

                const energy   = this._rms(chunk);
                const isSpeech = energy > this.SPEECH_THRESH;

                if (isSpeech && !this._hasSpeech) {
                    this._hasSpeech = true;
                    this._setIndicator("speaking");
                    this._setStatus("🔴 Recording… click Stop when done", "text-green-400");
                }

                // Pulse mic rings with volume level
                const ring1 = document.getElementById("mic-ring-1");
                const ring2 = document.getElementById("mic-ring-2");
                if (ring1) ring1.style.opacity = Math.min(energy * 20, 0.8).toFixed(2);
                if (ring2) ring2.style.opacity = Math.min(energy * 10, 0.4).toFixed(2);
            };

            // Connect: mic stream → worklet (no destination needed — we only read, not play)
            source.connect(this.workletNode);
            // Connecting to destination is NOT required and would cause echo.
            // The worklet keeps running as long as its input is connected.

            this.isRecording = true;
            this._watchdogTimer = setTimeout(() => {
                if (this.isRecording) { console.log("[SpeechCapture] Watchdog timeout"); this.stopRecording(); }
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
        if (this.workletNode) { this.workletNode.disconnect(); this.workletNode = null; }
        if (this.audioCtx)   { this.audioCtx.close();         this.audioCtx   = null; }
        if (this.stream)     { this.stream.getTracks().forEach(t => t.stop()); this.stream = null; }
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
        for (let i = 0; i < raw.length; i++)      { if (Math.abs(raw[i]) > this.SPEECH_THRESH * 0.5) { start = Math.max(0, i - padFrames); break; } }
        for (let i = raw.length - 1; i >= 0; i--) { if (Math.abs(raw[i]) > this.SPEECH_THRESH * 0.5) { end = Math.min(raw.length - 1, i + padFrames); break; } }

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
                transcript:             result.transcript             || "",
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
        if (this.workletNode) { this.workletNode.disconnect(); this.workletNode = null; }
        if (this.audioCtx)   { this.audioCtx.close().catch(() => {}); this.audioCtx = null; }
        if (this.stream)     { this.stream.getTracks().forEach(t => t.stop()); this.stream = null; }
        this._setIndicator("idle");
        this._showStartBtn("🎤 Start Recording");
        const status = document.getElementById("voice-status");
        if (status) status.textContent = "Press the mic button and say your phrase.";
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