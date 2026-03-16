// frontend/js/speech.js
// IMPROVED v2: better client-side noise detection + delta feature forwarding
//
// Key improvements:
//  1. Pre-recording noise floor estimation (100ms ambient measurement)
//  2. Dynamic voice threshold: max(VOICE_THRESHOLD, noiseFloor * 1.5)
//     → adapts to the room's background noise level before recording starts
//  3. Multi-band frequency check: validates that energy is in speech bands
//     (300–3400 Hz), not just broadband noise
//  4. Forwards delta_mfcc_mean, delta2_mfcc_mean, spectral_flux_mean,
//     voiced_fraction from the server response to the enrollment/auth flow
//  5. SNR feedback shown to the user in real-time
//  6. Recording auto-cancels if noise floor is too high before speech begins

const SpeechCapture = {
    mediaRecorder:  null,
    audioChunks:    [],
    isRecording:    false,
    currentAttempt: 1,
    maxAttempts:    3,

    // VAD
    audioContext:   null,
    analyser:       null,
    microphone:     null,
    javascriptNode: null,
    silenceStart:   null,
    voiced:         false,

    // Noise floor (estimated before recording)
    noiseFloor:     0.0,
    noiseEstimated: false,

    // Thresholds (dynamic — set after noise floor estimation)
    VOICE_THRESHOLD_BASE:  0.02,   // minimum; raised if room is noisy
    SILENCE_DURATION:      1200,   // ms of silence before stopping (was 1000)
    MIN_VOICE_DURATION:    2000,   // ms of voice needed (was 1500)
    MAX_NOISE_FLOOR:       0.08,   // above this → warn user about background noise
    SPEECH_BAND_MIN_RATIO: 0.15,   // min fraction of energy in 300-3400Hz band

    // Stream kept for noise estimation phase
    _stream: null,

    // ── Noise floor estimation ─────────────────────────────────────────────
    // Measures ambient noise for 100ms BEFORE recording starts.
    // Raises the voice trigger threshold accordingly.
    async _estimateNoiseFloor(stream) {
        return new Promise((resolve) => {
            const ctx      = new (window.AudioContext || window.webkitAudioContext)();
            const analyser = ctx.createAnalyser();
            const mic      = ctx.createMediaStreamSource(stream);
            analyser.fftSize              = 2048;
            analyser.smoothingTimeConstant = 0.0;  // no smoothing for measurement
            mic.connect(analyser);

            const samples = [];
            const MEASURE_MS = 150;
            const startTime  = Date.now();

            const measure = () => {
                const array = new Uint8Array(analyser.frequencyBinCount);
                analyser.getByteFrequencyData(array);
                let sum = 0;
                for (let i = 0; i < array.length; i++) sum += array[i] * array[i];
                samples.push(Math.sqrt(sum / array.length) / 255);

                if (Date.now() - startTime < MEASURE_MS) {
                    requestAnimationFrame(measure);
                } else {
                    mic.disconnect();
                    ctx.close();
                    const floor = samples.reduce((a, b) => a + b, 0) / samples.length;
                    resolve(floor);
                }
            };
            requestAnimationFrame(measure);
        });
    },

    // ── Check speech band energy ratio ────────────────────────────────────
    // Returns fraction of total spectral energy in 300–3400 Hz (speech band).
    // Background noise (fans, HVAC) tends to be broadband or very low-freq.
    _getSpeechBandRatio(analyser, sampleRate) {
        const array     = new Uint8Array(analyser.frequencyBinCount);
        analyser.getByteFrequencyData(array);
        const binHz     = sampleRate / (analyser.fftSize);
        const minBin    = Math.floor(300  / binHz);
        const maxBin    = Math.floor(3400 / binHz);

        let speechEnergy = 0, totalEnergy = 0;
        for (let i = 0; i < array.length; i++) {
            const e = array[i] * array[i];
            totalEnergy += e;
            if (i >= minBin && i <= maxBin) speechEnergy += e;
        }
        return totalEnergy > 0 ? speechEnergy / totalEnergy : 0;
    },

    async startRecording() {
        try {
            const stream = await navigator.mediaDevices.getUserMedia({
                audio: {
                    channelCount:     1,
                    echoCancellation: true,
                    noiseSuppression: true,
                    autoGainControl:  true,
                    sampleRate:       16000,
                }
            });
            this._stream = stream;

            // ── Step 1: Estimate noise floor ──────────────────────────────
            const status = document.getElementById("voice-status");
            if (status) {
                status.textContent = "🔇 Measuring background noise…";
                status.className   = "text-center text-sm mb-4 text-yellow-400";
            }

            this.noiseFloor     = await this._estimateNoiseFloor(stream);
            this.noiseEstimated = true;

            // Dynamic voice threshold: at least 1.5× noise floor, minimum 0.02
            const dynamicThreshold = Math.max(
                this.VOICE_THRESHOLD_BASE,
                this.noiseFloor * 1.8
            );
            this.VOICE_THRESHOLD = dynamicThreshold;

            console.log(`[SpeechCapture] Noise floor: ${(this.noiseFloor * 100).toFixed(1)}%  `
                      + `Voice threshold: ${(dynamicThreshold * 100).toFixed(1)}%`);

            // Warn if room is too noisy
            if (this.noiseFloor > this.MAX_NOISE_FLOOR) {
                if (status) {
                    status.textContent = (
                        `⚠️ High background noise detected (${(this.noiseFloor * 100).toFixed(0)}%). `
                        + "Please move to a quieter area and try again."
                    );
                    status.className = "text-center text-sm mb-4 text-red-400";
                }
                stream.getTracks().forEach(t => t.stop());
                const btn = document.getElementById("record-btn");
                if (btn) { btn.disabled = false; btn.textContent = "🎤 Try Again"; }
                return false;
            }

            if (status) {
                status.textContent = (
                    `✅ Background quiet (${(this.noiseFloor * 100).toFixed(0)}%). `
                    + "Speak now — recording starts when voice is detected…"
                );
                status.className = "text-center text-sm mb-4 text-green-400";
            }

            // ── Step 2: Set up VAD with dynamic threshold ─────────────────
            this.audioContext   = new (window.AudioContext || window.webkitAudioContext)();
            this.analyser       = this.audioContext.createAnalyser();
            this.microphone     = this.audioContext.createMediaStreamSource(stream);
            this.javascriptNode = this.audioContext.createScriptProcessor(2048, 1, 1);

            this.analyser.smoothingTimeConstant = 0.3;
            this.analyser.fftSize               = 2048;

            this.microphone.connect(this.analyser);
            this.analyser.connect(this.javascriptNode);
            this.javascriptNode.connect(this.audioContext.destination);

            const self            = this;
            let voiceDetectedTime = null;
            const sampleRate      = this.audioContext.sampleRate;

            this.javascriptNode.onaudioprocess = function () {
                const array = new Uint8Array(self.analyser.frequencyBinCount);
                self.analyser.getByteFrequencyData(array);

                let sum = 0;
                for (let i = 0; i < array.length; i++) sum += array[i] * array[i];
                const rms = Math.sqrt(sum / array.length) / 255;

                // Also check speech band ratio to avoid triggering on broadband noise
                const speechRatio = self._getSpeechBandRatio(self.analyser, sampleRate);

                self.updateVolumeIndicator(rms, speechRatio);

                const isVoice = rms > self.VOICE_THRESHOLD && speechRatio > self.SPEECH_BAND_MIN_RATIO;

                if (isVoice) {
                    if (!self.voiced) {
                        console.log(`🎤 Voice detected (rms=${(rms*100).toFixed(1)}% `
                                   + `speechBand=${(speechRatio*100).toFixed(0)}%)`);
                        self.voiced       = true;
                        voiceDetectedTime = Date.now();
                        if (!self.isRecording) self.startActualRecording(stream);
                    }
                    self.silenceStart = null;
                } else {
                    if (self.voiced) {
                        if (self.silenceStart === null) {
                            self.silenceStart = Date.now();
                        } else if (Date.now() - self.silenceStart > self.SILENCE_DURATION) {
                            const voiceDuration = self.silenceStart - voiceDetectedTime;
                            if (voiceDuration >= self.MIN_VOICE_DURATION) {
                                console.log(`✅ Voice complete (${voiceDuration}ms)`);
                                self.stopRecording();
                            } else {
                                console.log(`⚠️ Too short (${voiceDuration}ms), keep speaking`);
                                if (status) {
                                    status.textContent = "⚠️ Too short — keep speaking the full phrase";
                                    status.className   = "text-center text-sm mb-4 text-yellow-400";
                                }
                                self.silenceStart = null;
                                // Don't reset voiced — user may resume speaking
                            }
                        }
                    }
                }
            };

            return true;

        } catch (err) {
            console.error("Microphone error:", err);
            const status = document.getElementById("voice-status");
            if (status) status.textContent = "❌ Microphone access denied.";
            return false;
        }
    },

    startActualRecording(stream) {
        if (this.isRecording) return;

        this.audioChunks   = [];
        this.mediaRecorder = new MediaRecorder(stream);
        this.isRecording   = true;

        const status = document.getElementById("voice-status");
        if (status) {
            status.textContent = "🔴 Recording — speak the full phrase clearly…";
            status.className   = "text-center text-sm mb-4 text-red-400";
        }

        this.mediaRecorder.ondataavailable = (e) => {
            if (e.data.size > 0) this.audioChunks.push(e.data);
        };

        this.mediaRecorder.onstop = () => {
            const mimeType  = this.mediaRecorder.mimeType || "audio/webm";
            const audioBlob = new Blob(this.audioChunks, { type: mimeType });
            console.log("Recording stopped —", mimeType, audioBlob.size, "bytes");
            this.processAudio(audioBlob);
            stream.getTracks().forEach(t => t.stop());
            if (this.audioContext) this.audioContext.close();
        };

        this.mediaRecorder.start();
        setTimeout(() => {
            if (this.isRecording) {
                console.log("⏱ Max duration reached");
                this.stopRecording();
            }
        }, 12000);  // 12 second max (was 10)
    },

    stopRecording() {
        if (this.mediaRecorder && this.isRecording) {
            this.mediaRecorder.stop();
            this.isRecording = false;
            this.voiced      = false;
            if (this.javascriptNode) this.javascriptNode.disconnect();
            if (this.analyser)       this.analyser.disconnect();
            if (this.microphone)     this.microphone.disconnect();
        }
    },

    updateVolumeIndicator(level, speechRatio) {
        const indicator = document.getElementById("recording-indicator");
        if (!indicator) return;
        const isVoice = level > this.VOICE_THRESHOLD && speechRatio > this.SPEECH_BAND_MIN_RATIO;
        if (isVoice) {
            indicator.style.backgroundColor = "#10b981";
            indicator.classList.remove("animate-pulse");
        } else if (level > this.noiseFloor * 1.2) {
            // Audio detected but not classified as speech
            indicator.style.backgroundColor = "#f59e0b";
            indicator.classList.add("animate-pulse");
        } else {
            indicator.style.backgroundColor = "#ef4444";
            indicator.classList.add("animate-pulse");
        }
    },

    async processAudio(audioBlob) {
        const status = document.getElementById("voice-status");
        if (status) status.textContent = "⚙️ Processing audio…";

        const mimeType = audioBlob.type || "audio/webm";
        let format = "webm";
        if      (mimeType.includes("ogg")) format = "ogg";
        else if (mimeType.includes("mp4")) format = "mp4";
        else if (mimeType.includes("wav")) format = "wav";

        const base64Audio = await this.blobToBase64(audioBlob);
        const username    = typeof authUsername    !== "undefined" ? authUsername
                          : typeof currentUsername !== "undefined" ? currentUsername
                          : "";

        try {
            const response = await fetch(
                "http://127.0.0.1:8000/api/enroll/extract-mfcc", {
                method:  "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    audio_data:   base64Audio,
                    audio_format: format,
                    username,
                })
            });

            const result = await response.json();
            console.log("extract-mfcc result:", result);

            if (!result.success) {
                const errMsg = result.detail || "Try again";
                // Show SNR info if available
                const snrInfo = result.snr_db != null ? ` (SNR: ${result.snr_db.toFixed(1)}dB)` : "";
                if (status) {
                    status.textContent = `❌ ${errMsg}${snrInfo}`;
                    status.className   = "text-center text-sm mb-4 text-red-400";
                }
                const btn = document.getElementById("record-btn");
                if (btn) { btn.disabled = false; btn.textContent = "🎤 Try Again"; }
                return;
            }

            // Show SNR quality feedback to user
            if (result.snr_db != null && status && !status.textContent.startsWith("❌")) {
                const snrLabel = result.snr_db > 25 ? "excellent" :
                                 result.snr_db > 15 ? "good" : "acceptable";
                console.log(`Audio quality: ${snrLabel} (SNR=${result.snr_db.toFixed(1)}dB)`);
            }

            // ── Build FULL feature dict — all 62 features ─────────────────
            // NEW: includes delta_mfcc_mean, delta2_mfcc_mean, spectral_flux_mean,
            //      voiced_fraction — required for the improved 62-feature model.
            const fullFeatureDict = {
                mfcc_features:          result.mfcc_features          || [],
                mfcc_std:               result.mfcc_std               || [],
                delta_mfcc_mean:        result.delta_mfcc_mean        || [],   // NEW
                delta2_mfcc_mean:       result.delta2_mfcc_mean       || [],   // NEW
                pitch_mean:             result.pitch_mean             || 0,
                pitch_std:              result.pitch_std              || 0,
                speaking_rate:          result.speaking_rate          || 0,
                energy_mean:            result.energy_mean            || 0,
                energy_std:             result.energy_std             || 0,
                zcr_mean:               result.zcr_mean               || 0,
                spectral_centroid_mean: result.spectral_centroid_mean || 0,
                spectral_rolloff_mean:  result.spectral_rolloff_mean  || 0,
                spectral_flux_mean:     result.spectral_flux_mean     || 0,    // NEW
                voiced_fraction:        result.voiced_fraction        || 0,    // NEW
                snr_db:                 result.snr_db                 || 0,    // NEW (for logging)
            };

            if (typeof onVoiceAuthComplete === "function") {
                // ── LOGIN PAGE ───────────────────────────────────────────
                if (status) status.textContent = "⏳ Verifying voice…";
                onVoiceAuthComplete(fullFeatureDict);

            } else if (typeof onVoiceRecorded === "function") {
                // ── ENROLLMENT PAGE ──────────────────────────────────────
                if (status) {
                    status.textContent = `✅ Recording ${this.currentAttempt} processed!`;
                    status.className   = "text-center text-sm mb-4 text-green-400";
                }
                onVoiceRecorded(fullFeatureDict);
                this.currentAttempt++;

                const btn = document.getElementById("record-btn");
                if (btn) {
                    if (this.currentAttempt <= this.maxAttempts) {
                        btn.textContent = `🎤 Record Again (${this.currentAttempt}/${this.maxAttempts})`;
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
            console.error("Audio processing error:", err);
            if (status) {
                status.textContent = "❌ Could not connect to server.";
                status.className   = "text-center text-sm mb-4 text-red-400";
            }
        }
    },

    blobToBase64(blob) {
        return new Promise((resolve, reject) => {
            const reader     = new FileReader();
            reader.onloadend = () => resolve(reader.result.split(",")[1]);
            reader.onerror   = reject;
            reader.readAsDataURL(blob);
        });
    },

    reset() {
        this.audioChunks    = [];
        this.isRecording    = false;
        this.currentAttempt = 1;
        this.mediaRecorder  = null;
        this.voiced         = false;
        this.silenceStart   = null;
        this.noiseFloor     = 0.0;
        this.noiseEstimated = false;
        if (this.audioContext) {
            this.audioContext.close();
            this.audioContext = null;
        }
    }
};

function startRecording() {
    const btn       = document.getElementById("record-btn");
    const indicator = document.getElementById("recording-indicator");
    const status    = document.getElementById("voice-status");

    if (btn) {
        btn.disabled    = true;
        btn.textContent = "🎤 Measuring background…";
        btn.classList.replace("bg-red-600", "bg-yellow-600");
    }
    if (indicator) {
        indicator.classList.remove("hidden");
        indicator.style.backgroundColor = "#f59e0b";
    }
    if (status) {
        status.textContent = "🔇 Measuring background noise (stay quiet)…";
        status.className   = "text-center text-sm mb-4 text-yellow-400";
    }

    SpeechCapture.reset();
    SpeechCapture.startRecording().then(ok => {
        if (ok && btn) {
            btn.textContent = "🎤 Listening — speak the phrase…";
        }
    });
}
