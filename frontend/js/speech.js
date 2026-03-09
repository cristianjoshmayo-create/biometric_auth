// frontend/js/speech.js
// FIX: processAudio now passes the FULL feature dict (34 features) to auth,
//      not just mfcc_features (13 values).  The old code caused the model to
//      receive 21 zero-padded features on every auth attempt, making it
//      effectively blind to pitch, energy, ZCR, and spectral shape.

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

    // Thresholds
    VOICE_THRESHOLD:    0.02,
    SILENCE_DURATION:   1000,
    MIN_VOICE_DURATION: 1500,

    async startRecording() {
        try {
            const stream = await navigator.mediaDevices.getUserMedia({
                audio: {
                    channelCount:     1,
                    echoCancellation: true,
                    noiseSuppression: true,
                    autoGainControl:  true,
                }
            });

            this.audioContext  = new (window.AudioContext || window.webkitAudioContext)();
            this.analyser      = this.audioContext.createAnalyser();
            this.microphone    = this.audioContext.createMediaStreamSource(stream);
            this.javascriptNode = this.audioContext.createScriptProcessor(2048, 1, 1);

            this.analyser.smoothingTimeConstant = 0.3;
            this.analyser.fftSize = 1024;

            this.microphone.connect(this.analyser);
            this.analyser.connect(this.javascriptNode);
            this.javascriptNode.connect(this.audioContext.destination);

            const self = this;
            let voiceDetectedTime = null;

            this.javascriptNode.onaudioprocess = function () {
                const array = new Uint8Array(self.analyser.frequencyBinCount);
                self.analyser.getByteFrequencyData(array);

                let sum = 0;
                for (let i = 0; i < array.length; i++) sum += array[i] * array[i];
                const rms = Math.sqrt(sum / array.length) / 255;

                self.updateVolumeIndicator(rms);

                if (rms > self.VOICE_THRESHOLD) {
                    if (!self.voiced) {
                        console.log("🎤 Voice detected");
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
                                console.log(`⚠️ Too short (${voiceDuration}ms)`);
                                self.silenceStart = null;
                            }
                        }
                    }
                }
            };

            return true;

        } catch (err) {
            console.error("Microphone error:", err);
            document.getElementById("voice-status").textContent =
                "❌ Microphone access denied.";
            return false;
        }
    },

    startActualRecording(stream) {
        this.audioChunks  = [];
        this.mediaRecorder = new MediaRecorder(stream);
        this.isRecording  = true;

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
        }, 10000);
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

    updateVolumeIndicator(level) {
        const indicator = document.getElementById("recording-indicator");
        if (!indicator) return;
        if (level > this.VOICE_THRESHOLD) {
            indicator.style.backgroundColor = "#10b981";
            indicator.classList.remove("animate-pulse");
        } else {
            indicator.style.backgroundColor = "#ef4444";
            indicator.classList.add("animate-pulse");
        }
    },

    async processAudio(audioBlob) {
        const status = document.getElementById("voice-status");
        status.textContent = "⚙️ Processing audio…";

        const mimeType = audioBlob.type || "audio/webm";
        let format = "webm";
        if      (mimeType.includes("ogg")) format = "ogg";
        else if (mimeType.includes("mp4")) format = "mp4";
        else if (mimeType.includes("wav")) format = "wav";

        const base64Audio = await this.blobToBase64(audioBlob);
        const username    = typeof authUsername      !== "undefined" ? authUsername
                          : typeof currentUsername   !== "undefined" ? currentUsername
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
                status.textContent = "❌ " + (result.detail || "Try again");
                status.className   = "text-center text-sm mb-4 text-red-400";
                const btn = document.getElementById("record-btn");
                btn.disabled    = false;
                btn.textContent = "🎤 Try Again";
                return;
            }

            // ─────────────────────────────────────────────────────────────
            // FIX: build the FULL feature dict — all 34 features.
            //
            // OLD CODE (broken):
            //   onVoiceAuthComplete(result.mfcc_features)
            //   → only sent 13 MFCC means; model received 21 zeroes for
            //     pitch/energy/ZCR/spectral features → accepted everyone.
            //
            // NEW CODE: pass the complete result object so the backend
            // predict_voice() can build the proper 34-feature vector.
            // ─────────────────────────────────────────────────────────────
            const fullFeatureDict = {
                mfcc_features:          result.mfcc_features          || [],
                mfcc_std:               result.mfcc_std               || [],
                pitch_mean:             result.pitch_mean             || 0,
                pitch_std:              result.pitch_std              || 0,
                speaking_rate:          result.speaking_rate          || 0,
                energy_mean:            result.energy_mean            || 0,
                energy_std:             result.energy_std             || 0,
                zcr_mean:               result.zcr_mean               || 0,
                spectral_centroid_mean: result.spectral_centroid_mean || 0,
                spectral_rolloff_mean:  result.spectral_rolloff_mean  || 0,
            };

            if (typeof onVoiceAuthComplete === "function") {
                // ── LOGIN PAGE ──────────────────────────────────────────
                // Pass the full dict, not just mfcc_features
                status.textContent = "⏳ Verifying voice…";
                onVoiceAuthComplete(fullFeatureDict);

            } else if (typeof onVoiceRecorded === "function") {
                // ── ENROLLMENT PAGE ─────────────────────────────────────
                status.textContent = `✅ Recording ${this.currentAttempt} processed!`;
                status.className   = "text-center text-sm mb-4 text-green-400";
                // Enrollment can keep using fullFeatureDict too — future-proof
                onVoiceRecorded(fullFeatureDict);
                this.currentAttempt++;

                const btn = document.getElementById("record-btn");
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

        } catch (err) {
            console.error("Audio processing error:", err);
            status.textContent = "❌ Could not connect to server.";
            status.className   = "text-center text-sm mb-4 text-red-400";
        }
    },

    blobToBase64(blob) {
        return new Promise((resolve, reject) => {
            const reader   = new FileReader();
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

    btn.disabled    = true;
    btn.textContent = "🎤 Listening for voice…";
    btn.classList.replace("bg-red-600", "bg-yellow-600");
    indicator.classList.remove("hidden");
    indicator.style.backgroundColor = "#ef4444";
    status.textContent = "🎤 Speak now — recording starts when voice is detected";
    status.className   = "text-center text-sm mb-4 text-yellow-400";

    SpeechCapture.startRecording();
}