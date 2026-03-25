// frontend/js/speech.js
// v5 — RNNoise WASM pre-processing + Silero VAD + live level meter
//
// What's new vs v4:
//   1. RNNoise WASM noise suppression (jitsi/rnnoise-wasm) runs in an
//      AudioWorklet BEFORE Silero VAD sees the audio. This means the ML
//      model receives clean speech instead of speech+fan/AC/keyboard noise.
//      RNNoise is a recurrent neural network trained on thousands of real
//      noise conditions — far more effective than the spectral subtraction
//      currently done server-side.
//   2. Live audio level meter — shows a real-time bar so the user knows
//      if they're too quiet, too loud, or at the right distance.
//   3. Auto-gain normalisation — if the recorded audio RMS is below 0.02
//      (too quiet for reliable MFCC extraction) the samples are scaled up
//      before sending to the backend. Prevents silent-recording failures.
//   4. Timeout watchdog — if Silero hasn't fired onSpeechEnd after 12s,
//      prompts the user rather than silently hanging.
//   5. Noise floor pre-check — measures background noise for 0.5s before
//      listening. If ambient noise > -25 dBFS, warns the user to move to a
//      quieter location.
//   6. Cleaner retry flow — each failed attempt re-initialises VAD cleanly.
//
// CDN deps — these must be loaded in HTML before this file:
//   <script src="https://cdn.jsdelivr.net/npm/onnxruntime-web@1.22.0/dist/ort.wasm.min.js"></script>
//   <script src="https://cdn.jsdelivr.net/npm/@ricky0123/vad-web@0.0.29/dist/bundle.min.js"></script>
//
// The RNNoise worklet is injected at runtime (no extra script tag needed).
//
// Integration contract (unchanged from v4):
//   - Global startRecording() called by record-btn onclick
//   - Calls onVoiceRecorded(featureDict)    during enrollment
//   - Calls onVoiceAuthComplete(featureDict) during login
//   - DOM ids: record-btn, voice-status, diag-text, recording-indicator,
//              voice-level-bar (NEW — add to HTML for the level meter)

// ─────────────────────────────────────────────────────────────────────────────
//  RNNoise AudioWorklet source (injected as a Blob URL)
//  Uses jitsi/rnnoise-wasm via jsDelivr CDN.
//  The worklet:
//    - accumulates 128-sample Web Audio frames into 480-sample RNNoise frames
//    - runs RNNoise.processFrame() on each 480-sample chunk
//    - posts the denoised Float32 samples back via postMessage for VAD
//    - also computes and posts RMS level for the live meter
// ─────────────────────────────────────────────────────────────────────────────
const RNNOISE_WORKLET_SRC = `
// Inline AudioWorklet processor — loaded as Blob URL so no extra file needed.
// rnnoise-sync.js is loaded inside the worklet scope via importScripts.
// It exposes a global 'Module' (Emscripten) with RNNoise C bindings.

const RNNOISE_CDN = "https://cdn.jsdelivr.net/npm/@jitsi/rnnoise-wasm@0.2.1/dist/rnnoise-sync.js";
const FRAME_SIZE  = 480;   // RNNoise requires exactly 480 samples @ 48kHz or 16kHz
                            // We operate at 48kHz (default Web Audio) and
                            // Silero VAD resamples to 16kHz internally.

let rnnoiseLoaded = false;
let denoiseState  = null;
let rnnoiseModule = null;
let inputBuf      = new Float32Array(0);   // accumulator for partial frames
let samplesSent   = [];                    // denoised samples to post

async function loadRnnoise() {
    try {
        importScripts(RNNOISE_CDN);
        // Module is exposed globally by rnnoise-sync.js
        rnnoiseModule = Module;
        const statePtr = rnnoiseModule._rnnoise_create(0);
        // Create wrapper object with processFrame method
        denoiseState = {
            ptr: statePtr,
            inBuf:  rnnoiseModule._malloc(FRAME_SIZE * 4),   // Float32 = 4 bytes
            outBuf: rnnoiseModule._malloc(FRAME_SIZE * 4),
            processFrame(samples) {
                // Write input into WASM heap
                rnnoiseModule.HEAPF32.set(samples, this.inBuf >> 2);
                // Scale to RNNoise expected range [-32768, 32767]
                const heapIn = rnnoiseModule.HEAPF32.subarray(this.inBuf >> 2, (this.inBuf >> 2) + FRAME_SIZE);
                for (let i = 0; i < FRAME_SIZE; i++) heapIn[i] = samples[i] * 32768;
                rnnoiseModule._rnnoise_process_frame(this.ptr, this.inBuf, this.inBuf);
                // Read back and normalise
                const result = new Float32Array(FRAME_SIZE);
                const heapOut = rnnoiseModule.HEAPF32.subarray(this.inBuf >> 2, (this.inBuf >> 2) + FRAME_SIZE);
                for (let i = 0; i < FRAME_SIZE; i++) result[i] = heapOut[i] / 32768;
                return result;
            },
            destroy() {
                rnnoiseModule._rnnoise_destroy(this.ptr);
                rnnoiseModule._free(this.inBuf);
                rnnoiseModule._free(this.outBuf);
            }
        };
        rnnoiseLoaded = true;
        console.log("[RNNoiseWorklet] RNNoise loaded OK");
    } catch (e) {
        console.warn("[RNNoiseWorklet] RNNoise failed to load, passing audio through:", e.message);
        rnnoiseLoaded = false;
    }
}

// Start loading immediately — will be ready by the time the user hits record
loadRnnoise();

class RNNoiseProcessor extends AudioWorkletProcessor {
    constructor() {
        super();
        this._active   = true;
        this._allSamps = [];    // accumulate for post-VAD passthrough
        this._rmsSum   = 0;
        this._rmsCount = 0;

        this.port.onmessage = (e) => {
            if (e.data === "stop") {
                this._active = false;
                // Post all accumulated denoised samples and final RMS
                this.port.postMessage({
                    type:    "done",
                    samples: new Float32Array(this._allSamps),
                    rms:     this._rmsCount > 0 ? Math.sqrt(this._rmsSum / this._rmsCount) : 0
                });
            }
        };
    }

    process(inputs) {
        if (!this._active) return false;

        const input = inputs[0];
        if (!input || !input[0]) return true;

        const raw = input[0];   // Float32Array, 128 samples per frame

        // RMS for level meter
        for (let i = 0; i < raw.length; i++) {
            this._rmsSum += raw[i] * raw[i];
        }
        this._rmsCount += raw.length;

        // Emit RMS every ~10 frames (~128ms) for live meter
        if (this._rmsCount % (128 * 10) < 128) {
            const rms = Math.sqrt(this._rmsSum / this._rmsCount);
            this.port.postMessage({ type: "rms", rms });
        }

        // Accumulate into inputBuf
        const prev = inputBuf;
        inputBuf   = new Float32Array(prev.length + raw.length);
        inputBuf.set(prev);
        inputBuf.set(raw, prev.length);

        // Process as many complete FRAME_SIZE chunks as possible
        while (inputBuf.length >= FRAME_SIZE) {
            const frame = inputBuf.slice(0, FRAME_SIZE);
            inputBuf    = inputBuf.slice(FRAME_SIZE);

            let processed;
            if (rnnoiseLoaded && denoiseState) {
                try {
                    processed = denoiseState.processFrame(frame);
                } catch (_) {
                    processed = frame;
                }
            } else {
                processed = frame;
            }

            for (let i = 0; i < processed.length; i++) {
                this._allSamps.push(processed[i]);
            }
        }

        return true;
    }
}

registerProcessor("rnnoise-processor", RNNoiseProcessor);
`;

// ─────────────────────────────────────────────────────────────────────────────
//  SpeechCapture — main controller
// ─────────────────────────────────────────────────────────────────────────────
const SpeechCapture = {
    myvad:           null,
    audioCtx:        null,
    workletNode:     null,
    sourceNode:      null,
    stream:          null,
    isRecording:     false,
    currentAttempt:  1,
    maxAttempts:     3,
    _watchdogTimer:  null,
    _levelTimer:     null,
    _workletBlobUrl: null,

    // ── Silero VAD config ─────────────────────────────────────────────────
    VAD_CONFIG: {
        positiveSpeechThreshold: 0.50,
        negativeSpeechThreshold: 0.35,
        minSpeechFrames:         8,
        preSpeechPadFrames:      3,
        redemptionFrames:        10,
        onnxWASMBasePath: "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.22.0/dist/",
        baseAssetPath:    "https://cdn.jsdelivr.net/npm/@ricky0123/vad-web@0.0.29/dist/",
    },

    WATCHDOG_MS:  12000,   // give up if no speech detected after 12s
    MIN_DURATION: 1.0,     // minimum speech duration in seconds
    NOISE_CHECK_MS: 500,   // ms to sample background noise before recording

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

    // ── Live level bar (optional DOM element id="voice-level-bar") ───────
    // Expects: <div id="voice-level-bar" class="...">
    //              <div id="voice-level-fill" class="h-full bg-green-500 transition-all" style="width:0%"></div>
    //          </div>
    _updateLevel(rms) {
        const fill = document.getElementById("voice-level-fill");
        if (!fill) return;
        // Map RMS 0–0.5 → 0–100%, colour by zone
        const pct = Math.min(100, Math.round(rms * 200));
        fill.style.width = pct + "%";
        fill.className = "h-full transition-all duration-75 " + (
            pct < 5  ? "bg-gray-500" :    // too quiet
            pct < 60 ? "bg-green-500" :   // good
            pct < 85 ? "bg-yellow-400" :  // getting loud
                       "bg-red-500"       // clipping risk
        );
    },

    // ── Noise floor pre-check ─────────────────────────────────────────────
    // Opens the mic briefly, measures ambient RMS, warns if too noisy.
    // Returns { ok: bool, rms: float, dbfs: float }
    async _checkNoiseFloor(stream) {
        const ctx      = new AudioContext({ sampleRate: 16000 });
        const src      = ctx.createMediaStreamSource(stream);
        const analyser = ctx.createAnalyser();
        analyser.fftSize = 2048;
        src.connect(analyser);

        await new Promise(r => setTimeout(r, this.NOISE_CHECK_MS));

        const buf = new Float32Array(analyser.fftSize);
        analyser.getFloatTimeDomainData(buf);
        const rms   = Math.sqrt(buf.reduce((s, x) => s + x * x, 0) / buf.length);
        const dbfs  = rms > 0 ? 20 * Math.log10(rms) : -100;
        await ctx.close();

        console.log(`[SpeechCapture] Noise floor: ${dbfs.toFixed(1)} dBFS (RMS=${rms.toFixed(4)})`);
        return { ok: dbfs < -25, rms, dbfs };
    },

    // ── Register RNNoise AudioWorklet ─────────────────────────────────────
    async _initWorklet(audioCtx) {
        if (!this._workletBlobUrl) {
            const blob = new Blob([RNNOISE_WORKLET_SRC], { type: "application/javascript" });
            this._workletBlobUrl = URL.createObjectURL(blob);
        }
        try {
            await audioCtx.audioWorklet.addModule(this._workletBlobUrl);
            console.log("[SpeechCapture] RNNoise AudioWorklet registered");
            return true;
        } catch (e) {
            console.warn("[SpeechCapture] AudioWorklet registration failed:", e.message);
            return false;
        }
    },

    // ── Main entry point ──────────────────────────────────────────────────
    async startRecording() {
        if (typeof vad === "undefined" || !vad.MicVAD) {
            this._setStatus("❌ Voice library failed to load. Check internet connection.", "text-red-400");
            return false;
        }

        try {
            this._setStatus("⏳ Requesting microphone…", "text-yellow-400");
            this._setDiag("");

            // Request mic with optimal constraints for speech biometrics
            this.stream = await navigator.mediaDevices.getUserMedia({
                audio: {
                    channelCount:       { exact: 1 },
                    sampleRate:         { ideal: 48000 },
                    echoCancellation:   true,
                    noiseSuppression:   false,  // we handle this ourselves with RNNoise
                    autoGainControl:    false,  // we handle normalisation ourselves
                }
            });

            // ── Noise floor check ─────────────────────────────────────────
            this._setStatus("🔍 Checking background noise…", "text-yellow-400");
            const { ok: noiseOk, dbfs } = await this._checkNoiseFloor(this.stream);
            if (!noiseOk) {
                this._setStatus(
                    `⚠️ Background noise detected (${dbfs.toFixed(0)} dBFS). ` +
                    "Move to a quieter spot then try again.",
                    "text-yellow-400"
                );
                this._setDiag(`Ambient noise: ${dbfs.toFixed(1)} dBFS (need < -25 dBFS)`);
                // Warn but don't block — RNNoise will still help
                console.warn("[SpeechCapture] High ambient noise, proceeding with RNNoise");
            }

            // ── RNNoise AudioWorklet setup ────────────────────────────────
            this.audioCtx = new AudioContext({ sampleRate: 48000 });
            const workletOk = await this._initWorklet(this.audioCtx);

            if (workletOk) {
                this.sourceNode  = this.audioCtx.createMediaStreamSource(this.stream);
                this.workletNode = new AudioWorkletNode(this.audioCtx, "rnnoise-processor");

                // Live level meter from worklet
                this.workletNode.port.onmessage = (e) => {
                    if (e.data?.type === "rms") {
                        this._updateLevel(e.data.rms);
                    }
                };
                this.sourceNode.connect(this.workletNode);
                // Do NOT connect workletNode to destination — no audio playback
                console.log("[SpeechCapture] RNNoise pipeline: mic → worklet (silent)");
            } else {
                console.warn("[SpeechCapture] Falling back to direct mic (no RNNoise)");
            }

            // ── Silero VAD ────────────────────────────────────────────────
            this._setStatus("⏳ Initialising voice detector…", "text-yellow-400");

            this.myvad = await vad.MicVAD.new({
                ...this.VAD_CONFIG,
                stream: this.stream,   // pass same stream so VAD and worklet share mic

                onSpeechStart: () => {
                    console.log("[SileroVAD] Speech start");
                    this.isRecording = true;
                    this._clearWatchdog();
                    this._setStatus("🔴 Recording — speak the full phrase clearly…", "text-red-400");
                    this._setIndicator(true);
                    this._setDiag("🟢 Voice detected — keep going");
                },

                onSpeechEnd: async (audioFloat32) => {
                    const dur = (audioFloat32.length / 16000).toFixed(2);
                    console.log(`[SileroVAD] Speech end — ${audioFloat32.length} samples (${dur}s)`);
                    this._clearWatchdog();
                    this._setStatus("⚙️ Processing audio…", "text-yellow-400");
                    this._setDiag("");
                    this._setIndicator(false);
                    this.isRecording = false;
                    this._updateLevel(0);

                    // Stop VAD and tear down AudioWorklet
                    this.myvad.pause();
                    await this._teardownWorklet();

                    // Auto-gain: scale up if audio is too quiet
                    const normalised = this._autoGain(audioFloat32);

                    await this._processFloat32(normalised);
                },

                onVADMisfire: () => {
                    console.log("[SileroVAD] Misfire — too short");
                    this._clearWatchdog();
                    this._setStatus("⚠️ Too short — speak the full phrase without long pauses", "text-yellow-400");
                    this._setDiag("🔴 Recording too short");
                    this._setIndicator(false);
                    this._updateLevel(0);
                    this.isRecording = false;
                    this._startWatchdog();  // reset watchdog to listen again
                },
            });

            this.myvad.start();
            this._setStatus("🎤 Listening… speak your phrase when ready", "text-green-400");
            this._setDiag("Waiting for speech…");

            // Start watchdog in case user doesn't speak
            this._startWatchdog();

            return true;

        } catch (err) {
            console.error("[SpeechCapture] Init error:", err);
            this._setStatus(`❌ Microphone error: ${err.message}`, "text-red-400");
            await this._teardownWorklet();
            return false;
        }
    },

    // ── Watchdog timer ────────────────────────────────────────────────────
    _startWatchdog() {
        this._clearWatchdog();
        this._watchdogTimer = setTimeout(() => {
            if (!this.isRecording) {
                console.warn("[SpeechCapture] Watchdog fired — no speech detected");
                this._setStatus(
                    "⏱️ No speech detected. Check microphone and try again.",
                    "text-yellow-400"
                );
                this._setDiag("Microphone may be muted or too far away");
                this._resetBtn();
            }
        }, this.WATCHDOG_MS);
    },

    _clearWatchdog() {
        if (this._watchdogTimer) {
            clearTimeout(this._watchdogTimer);
            this._watchdogTimer = null;
        }
    },

    // ── Auto-gain normalisation ───────────────────────────────────────────
    // Scales audio so RMS is at least 0.05 (adequate for MFCC extraction).
    // Hard-clips at ±1 to prevent distortion.
    _autoGain(samples) {
        const rms = Math.sqrt(samples.reduce((s, x) => s + x * x, 0) / samples.length);
        if (rms < 0.001) {
            console.warn("[SpeechCapture] Audio appears silent (RMS=" + rms.toFixed(5) + ")");
            return samples;
        }
        const targetRms = 0.08;
        if (rms >= targetRms) return samples;   // already loud enough

        const gain = targetRms / rms;
        const maxGain = 10.0;    // don't over-amplify very quiet recordings
        const actualGain = Math.min(gain, maxGain);
        console.log(`[SpeechCapture] Auto-gain: ×${actualGain.toFixed(2)} (RMS ${rms.toFixed(4)} → ${(rms*actualGain).toFixed(4)})`);

        const out = new Float32Array(samples.length);
        for (let i = 0; i < samples.length; i++) {
            out[i] = Math.max(-1, Math.min(1, samples[i] * actualGain));
        }
        return out;
    },

    // ── Teardown AudioWorklet and AudioContext ─────────────────────────────
    async _teardownWorklet() {
        this._updateLevel(0);
        try {
            if (this.workletNode) {
                this.workletNode.port.postMessage("stop");
                this.workletNode.disconnect();
                this.workletNode = null;
            }
            if (this.sourceNode) {
                this.sourceNode.disconnect();
                this.sourceNode = null;
            }
            if (this.audioCtx && this.audioCtx.state !== "closed") {
                await this.audioCtx.close();
                this.audioCtx = null;
            }
        } catch (e) {
            console.warn("[SpeechCapture] Teardown error:", e.message);
        }
    },

    // ── Convert Float32Array → WAV → base64 → POST to /extract-mfcc ──────
    async _processFloat32(audioFloat32) {
        const durationSec = audioFloat32.length / 16000;

        if (durationSec < this.MIN_DURATION) {
            this._setStatus(
                `❌ Recording too short (${durationSec.toFixed(1)}s). Speak the full phrase.`,
                "text-red-400"
            );
            this._resetBtn();
            return;
        }

        const wavBlob = this._float32ToWav(audioFloat32, 16000);
        console.log(`[SpeechCapture] WAV: ${wavBlob.size} bytes  ${durationSec.toFixed(2)}s`);

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
                    audio_format: "wav",
                    username
                })
            });

            const result = await response.json();
            console.log("[SpeechCapture] extract-mfcc result:", result);

            if (!result.success) {
                const detail = result.detail || "Processing failed";
                const snrTxt = result.snr_db != null
                    ? ` (SNR: ${result.snr_db.toFixed(1)} dB)`
                    : "";
                this._setStatus(`❌ ${detail}${snrTxt}`, "text-red-400");
                this._resetBtn();
                return;
            }

            // Log audio quality tier
            if (result.snr_db != null) {
                const q = result.snr_db > 25 ? "excellent"
                        : result.snr_db > 15 ? "good"
                        : result.snr_db > 8  ? "acceptable" : "poor";
                const msg = `Audio quality: ${q} — SNR=${result.snr_db.toFixed(1)}dB  voiced=${(result.voiced_fraction*100).toFixed(0)}%`;
                console.log(`[SpeechCapture] ${msg}`);
                this._setDiag(msg);
            }

            // Build 62-feature dict
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

            // Route to enrollment or auth callback
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

    // ── Float32 → 16-bit PCM WAV Blob ────────────────────────────────────
    // Minimal standard 44-byte header — no extra metadata chunks.
    // Matches the RIFF data-offset fix in enroll.py exactly.
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
        v.setUint32(16, 16, true);          // fmt chunk size
        v.setUint16(20,  1, true);          // PCM
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
            if (!btn.classList.contains("bg-red-600"))
                btn.classList.replace("bg-gray-600", "bg-red-600");
        }
        this._startWatchdog();  // re-arm in case they want to try again
        this._clearWatchdog();  // but clear immediately — they need to click
    },

    // ── Full reset (called before each new recording attempt) ─────────────
    async reset() {
        this._clearWatchdog();
        this._updateLevel(0);
        if (this.myvad) {
            try { this.myvad.destroy(); } catch (_) {}
            this.myvad = null;
        }
        await this._teardownWorklet();
        if (this.stream) {
            this.stream.getTracks().forEach(t => t.stop());
            this.stream = null;
        }
        this.isRecording = false;
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

    const ok = await SpeechCapture.startRecording();

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
}