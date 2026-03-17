// frontend/js/speech.js
// v3 — DIAGNOSTIC + LENIENT THRESHOLDS
//
// Root cause of "can't recognise I'm speaking":
//   6 stacked rejection gates were all calibrated for a professional quiet room.
//   On a normal laptop mic in a typical room, recordings were silently blocked
//   before ever reaching the ML model.
//
// Changes in v3:
//   1. MAX_NOISE_FLOOR      0.08  → 0.20   (most rooms have fans/AC)
//   2. VOICE_THRESHOLD_BASE 0.02  → 0.01   (laptop mics are quieter than expected)
//   3. noiseFloor multiplier 1.8x → 1.3x   (less aggressive dynamic threshold)
//   4. SPEECH_BAND_MIN_RATIO 0.15 → 0.08   (softened — speech band check was too strict)
//   5. MIN_VOICE_DURATION   2000ms → 1500ms (shorter phrase is still enough)
//   6. SILENCE_DURATION     1200ms → 1500ms (more forgiving pause detection)
//   7. Diagnostic bar shows live RMS %, speech band %, and threshold — user
//      can now SEE why their voice isn't triggering
//   8. Server-side gates also relaxed (see enroll.py): SNR 10→6, voiced 60%→45%

const SpeechCapture = {
    mediaRecorder:  null,
    audioChunks:    [],
    isRecording:    false,
    currentAttempt: 1,
    maxAttempts:    3,

    // VAD state
    audioContext:   null,
    analyser:       null,
    microphone:     null,
    javascriptNode: null,
    silenceStart:   null,
    voiced:         false,

    // Noise floor measured before recording
    noiseFloor:     0.0,
    noiseEstimated: false,

    // ── Thresholds — calibrated for real-world laptop/phone mics ─────────
    VOICE_THRESHOLD_BASE:  0.01,   // lowered: laptop mics output lower RMS than pro mics
    SILENCE_DURATION:      1500,   // ms silence before auto-stop (more forgiving)
    MIN_VOICE_DURATION:    1500,   // ms of speech required (shorter phrase ok)
    MAX_NOISE_FLOOR:       0.20,   // raised: fans/AC/office noise is normal
    SPEECH_BAND_MIN_RATIO: 0.08,   // lowered: compressed audio formats reduce band ratio

    _stream: null,

    // ── Noise floor estimation (150ms ambient sample) ──────────────────────
    async _estimateNoiseFloor(stream) {
        return new Promise((resolve) => {
            const ctx      = new (window.AudioContext || window.webkitAudioContext)();
            const analyser = ctx.createAnalyser();
            const mic      = ctx.createMediaStreamSource(stream);
            analyser.fftSize               = 2048;
            analyser.smoothingTimeConstant = 0.0;
            mic.connect(analyser);

            const samples   = [];
            const startTime = Date.now();

            const measure = () => {
                const array = new Uint8Array(analyser.frequencyBinCount);
                analyser.getByteFrequencyData(array);
                let sum = 0;
                for (let i = 0; i < array.length; i++) sum += array[i] * array[i];
                samples.push(Math.sqrt(sum / array.length) / 255);

                if (Date.now() - startTime < 150) {
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

    // ── Speech band energy ratio (300–3400 Hz) ─────────────────────────────
    _getSpeechBandRatio(analyser, sampleRate) {
        const array  = new Uint8Array(analyser.frequencyBinCount);
        analyser.getByteFrequencyData(array);
        const binHz  = sampleRate / analyser.fftSize;
        const minBin = Math.floor(300  / binHz);
        const maxBin = Math.floor(3400 / binHz);

        let speechEnergy = 0, totalEnergy = 0;
        for (let i = 0; i < array.length; i++) {
            const e = array[i] * array[i];
            totalEnergy += e;
            if (i >= minBin && i <= maxBin) speechEnergy += e;
        }
        return totalEnergy > 0 ? speechEnergy / totalEnergy : 0;
    },

    // ── Update the live diagnostic bar ────────────────────────────────────
    // Shows RMS level, speech band %, and whether voice is detected.
    // This is the main debugging tool — users can see exactly why voice
    // isn't triggering instead of getting a silent failure.
    _updateDiagBar(rms, speechRatio, isVoice) {
        const bar     = document.getElementById("diag-bar");
        const diagTxt = document.getElementById("diag-text");
        if (!bar && !diagTxt) return;

        const rmsPC    = (rms * 100).toFixed(1);
        const bandPC   = (speechRatio * 100).toFixed(0);
        const threshPC = (this.VOICE_THRESHOLD * 100).toFixed(1);

        if (diagTxt) {
            diagTxt.textContent = isVoice
                ? `🟢 Voice detected  (level: ${rmsPC}%  band: ${bandPC}%)`
                : `🔴 No voice  (level: ${rmsPC}% / need ${threshPC}%  band: ${bandPC}% / need ${(this.SPEECH_BAND_MIN_RATIO*100).toFixed(0)}%)`;
        }

        if (bar) {
            const fillPct = Math.min(100, (rms / (this.VOICE_THRESHOLD * 2)) * 100);
            bar.style.width      = `${fillPct}%`;
            bar.style.background = isVoice ? "#10b981" : rms > this.noiseFloor * 1.1 ? "#f59e0b" : "#ef4444";
        }

        // Also update the existing recording indicator if present
        const indicator = document.getElementById("recording-indicator");
        if (indicator) {
            indicator.style.backgroundColor = isVoice ? "#10b981" : rms > this.noiseFloor * 1.1 ? "#f59e0b" : "#ef4444";
            if (isVoice) indicator.classList.remove("animate-pulse");
            else indicator.classList.add("animate-pulse");
        }
    },

    // ── Main entry point ──────────────────────────────────────────────────
    async startRecording() {
        const status = document.getElementById("voice-status");

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

            // Step 1 — measure noise floor
            if (status) {
                status.textContent = "🔇 Measuring background noise…";
                status.className   = "text-center text-sm mb-4 text-yellow-400";
            }

            this.noiseFloor     = await this._estimateNoiseFloor(stream);
            this.noiseEstimated = true;

            // Dynamic threshold: 1.3× noise floor, minimum 0.01
            // (was 1.8×, which was too aggressive — any ambient noise pushed
            //  the threshold above what a laptop mic can produce for speech)
            const dynamicThreshold = Math.max(this.VOICE_THRESHOLD_BASE, this.noiseFloor * 1.3);
            this.VOICE_THRESHOLD   = dynamicThreshold;

            console.log(
                `[SpeechCapture] noiseFloor=${(this.noiseFloor*100).toFixed(1)}%  ` +
                `voiceThreshold=${(dynamicThreshold*100).toFixed(1)}%  ` +
                `maxAllowed=${(this.MAX_NOISE_FLOOR*100).toFixed(0)}%`
            );

            // Warn if room is extremely noisy — but much higher bar than before
            if (this.noiseFloor > this.MAX_NOISE_FLOOR) {
                if (status) {
                    status.textContent =
                        `⚠️ Very loud background noise (${(this.noiseFloor*100).toFixed(0)}%). ` +
                        "Try moving away from fans or speakers.";
                    status.className = "text-center text-sm mb-4 text-yellow-400";
                    // Warn but DON'T abort — let them try anyway
                }
            } else if (status) {
                status.textContent =
                    `✅ Ready (noise: ${(this.noiseFloor*100).toFixed(0)}%). ` +
                    "Speak now — recording starts automatically…";
                status.className = "text-center text-sm mb-4 text-green-400";
            }

            // Step 2 — set up VAD
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

                const speechRatio = self._getSpeechBandRatio(self.analyser, sampleRate);
                const isVoice     = rms > self.VOICE_THRESHOLD && speechRatio > self.SPEECH_BAND_MIN_RATIO;

                self._updateDiagBar(rms, speechRatio, isVoice);

                if (isVoice) {
                    if (!self.voiced) {
                        console.log(`🎤 Voice detected — rms=${(rms*100).toFixed(1)}%  band=${(speechRatio*100).toFixed(0)}%  threshold=${(self.VOICE_THRESHOLD*100).toFixed(1)}%`);
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
                                console.log(`✅ Voice segment complete: ${voiceDuration}ms`);
                                self.stopRecording();
                            } else {
                                // Too short — keep listening rather than failing
                                console.log(`⚠️ Voice too short (${voiceDuration}ms < ${self.MIN_VOICE_DURATION}ms) — keep speaking`);
                                if (status) {
                                    status.textContent = `⚠️ Keep speaking — need ${(self.MIN_VOICE_DURATION/1000).toFixed(1)}s of voice`;
                                    status.className   = "text-center text-sm mb-4 text-yellow-400";
                                }
                                self.silenceStart = null;
                                // Reset voiced so they can start a fresh attempt at the phrase
                                self.voiced = false;
                            }
                        }
                    }
                }
            };

            return true;

        } catch (err) {
            console.error("Microphone error:", err);
            if (status) {
                status.textContent = `❌ Microphone error: ${err.message}`;
                status.className   = "text-center text-sm mb-4 text-red-400";
            }
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
            console.log(`Recording stopped — ${mimeType}  ${audioBlob.size} bytes`);
            this.processAudio(audioBlob);
            stream.getTracks().forEach(t => t.stop());
            if (this.audioContext) this.audioContext.close();
        };

        this.mediaRecorder.start();

        // 15 second hard cap (generous — the phrase takes ~4s)
        setTimeout(() => {
            if (this.isRecording) {
                console.log("⏱ Max duration reached, auto-stopping");
                this.stopRecording();
            }
        }, 15000);
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

    async processAudio(audioBlob) {
        const status = document.getElementById("voice-status");
        const diagTxt = document.getElementById("diag-text");
        if (status) status.textContent = "⚙️ Processing audio…";
        if (diagTxt) diagTxt.textContent = "";

        const mimeType = audioBlob.type || "audio/webm";
        let format = "webm";
        if      (mimeType.includes("ogg")) format = "ogg";
        else if (mimeType.includes("mp4")) format = "mp4";
        else if (mimeType.includes("wav")) format = "wav";

        // Basic size sanity check before sending
        if (audioBlob.size < 3000) {
            if (status) {
                status.textContent = "❌ Recording too short or silent. Speak louder and try again.";
                status.className   = "text-center text-sm mb-4 text-red-400";
            }
            const btn = document.getElementById("record-btn");
            if (btn) { btn.disabled = false; btn.textContent = "🎤 Try Again"; }
            return;
        }

        const base64Audio = await this.blobToBase64(audioBlob);
        const username    = typeof authUsername    !== "undefined" ? authUsername
                          : typeof currentUsername !== "undefined" ? currentUsername
                          : "";

        try {
            const response = await fetch(`${API_BASE}/enroll/extract-mfcc`, {
                method:  "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ audio_data: base64Audio, audio_format: format, username })
            });

            const result = await response.json();
            console.log("extract-mfcc result:", result);

            if (!result.success) {
                // Show the exact rejection reason so the user knows what to fix
                const detail = result.detail || "Processing failed";
                const snrTxt = result.snr_db != null ? ` (SNR: ${result.snr_db.toFixed(1)} dB)` : "";
                if (status) {
                    status.textContent = `❌ ${detail}${snrTxt}`;
                    status.className   = "text-center text-sm mb-4 text-red-400";
                }
                const btn = document.getElementById("record-btn");
                if (btn) { btn.disabled = false; btn.textContent = "🎤 Try Again"; }
                return;
            }

            // Show quality info in console for debugging
            if (result.snr_db != null) {
                const q = result.snr_db > 25 ? "excellent" : result.snr_db > 15 ? "good" : result.snr_db > 8 ? "acceptable" : "poor";
                console.log(`✅ Audio quality: ${q} (SNR=${result.snr_db.toFixed(1)}dB  voiced=${(result.voiced_fraction*100).toFixed(0)}%)`);
            }

            // Build full 62-feature dict
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

            if (typeof onVoiceAuthComplete === "function") {
                if (status) status.textContent = "⏳ Verifying voice…";
                onVoiceAuthComplete(fullFeatureDict);

            } else if (typeof onVoiceRecorded === "function") {
                if (status) {
                    status.textContent = `✅ Recording ${this.currentAttempt} saved!`;
                    status.className   = "text-center text-sm mb-4 text-green-400";
                }
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
    const diagTxt   = document.getElementById("diag-text");

    if (btn) {
        btn.disabled    = true;
        btn.textContent = "🎤 Measuring…";
        btn.classList.replace("bg-red-600", "bg-yellow-600");
    }
    if (indicator) {
        indicator.classList.remove("hidden");
        indicator.style.backgroundColor = "#f59e0b";
    }
    if (status) {
        status.textContent = "🔇 Stay quiet for a moment…";
        status.className   = "text-center text-sm mb-4 text-yellow-400";
    }
    if (diagTxt) diagTxt.textContent = "";

    SpeechCapture.reset();
    SpeechCapture.startRecording().then(ok => {
        if (ok && btn) {
            btn.textContent = "🎤 Listening…";
            btn.classList.replace("bg-yellow-600", "bg-red-600");
        } else if (!ok) {
            if (btn) {
                btn.disabled    = false;
                btn.textContent = "🎤 Try Again";
                btn.classList.replace("bg-yellow-600", "bg-red-600");
            }
        }
    });
}