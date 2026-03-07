// frontend/js/speech.js
// Voice recording with REAL-TIME Voice Activity Detection

const SpeechCapture = {
    mediaRecorder: null,
    audioChunks: [],
    isRecording: false,
    currentAttempt: 1,
    maxAttempts: 3,
    
    // VAD parameters
    audioContext: null,
    analyser: null,
    microphone: null,
    javascriptNode: null,
    silenceStart: null,
    voiced: false,
    
    // Thresholds
    VOICE_THRESHOLD: 0.02,      // RMS threshold for voice detection
    SILENCE_DURATION: 1000,      // 1 second of silence = stop recording
    MIN_VOICE_DURATION: 1500,    // Need at least 1.5 seconds of voice

    async startRecording() {
        try {
            const stream = await navigator.mediaDevices.getUserMedia({ 
                audio: {
                    channelCount: 1,
                    echoCancellation: true,
                    noiseSuppression: true,
                    autoGainControl: true  // Normalize volume
                }
            });

            // Set up audio analysis
            this.audioContext = new (window.AudioContext || window.webkitAudioContext)();
            this.analyser = this.audioContext.createAnalyser();
            this.microphone = this.audioContext.createMediaStreamSource(stream);
            this.javascriptNode = this.audioContext.createScriptProcessor(2048, 1, 1);

            this.analyser.smoothingTimeConstant = 0.3;
            this.analyser.fftSize = 1024;

            this.microphone.connect(this.analyser);
            this.analyser.connect(this.javascriptNode);
            this.javascriptNode.connect(this.audioContext.destination);

            // Real-time audio level monitoring
            const self = this;
            let voiceDetectedTime = null;
            
            this.javascriptNode.onaudioprocess = function() {
                const array = new Uint8Array(self.analyser.frequencyBinCount);
                self.analyser.getByteFrequencyData(array);
                
                // Calculate RMS (volume level)
                let sum = 0;
                for (let i = 0; i < array.length; i++) {
                    sum += array[i] * array[i];
                }
                const rms = Math.sqrt(sum / array.length) / 255;

                // Update status indicator
                self.updateVolumeIndicator(rms);

                // Voice activity detection
                if (rms > self.VOICE_THRESHOLD) {
                    // Voice detected
                    if (!self.voiced) {
                        console.log("🎤 Voice detected, starting recording...");
                        self.voiced = true;
                        voiceDetectedTime = Date.now();
                        
                        // Start actual MediaRecorder
                        if (!self.isRecording) {
                            self.startActualRecording(stream);
                        }
                    }
                    self.silenceStart = null;
                    
                } else {
                    // Silence detected
                    if (self.voiced) {
                        if (self.silenceStart === null) {
                            self.silenceStart = Date.now();
                        } else {
                            const silenceDuration = Date.now() - self.silenceStart;
                            
                            // Stop if silence too long
                            if (silenceDuration > self.SILENCE_DURATION) {
                                const voiceDuration = self.silenceStart - voiceDetectedTime;
                                
                                if (voiceDuration >= self.MIN_VOICE_DURATION) {
                                    console.log(`✅ Voice recording complete (${voiceDuration}ms)`);
                                    self.stopRecording();
                                } else {
                                    console.log(`⚠️ Voice too short (${voiceDuration}ms), continuing...`);
                                    self.silenceStart = null;
                                }
                            }
                        }
                    }
                }
            };

            return true;

        } catch (err) {
            console.error("Microphone error:", err);
            document.getElementById("voice-status").textContent = 
                "❌ Microphone access denied. Please allow microphone.";
            return false;
        }
    },

    startActualRecording(stream) {
        this.audioChunks = [];
        this.mediaRecorder = new MediaRecorder(stream);
        this.isRecording = true;

        this.mediaRecorder.ondataavailable = (event) => {
            if (event.data.size > 0) {
                this.audioChunks.push(event.data);
            }
        };

        this.mediaRecorder.onstop = () => {
            const mimeType = this.mediaRecorder.mimeType || 'audio/webm';
            const audioBlob = new Blob(this.audioChunks, { type: mimeType });
            
            console.log("Recording mime type:", mimeType);
            console.log("Audio blob size:", audioBlob.size);
            
            this.processAudio(audioBlob);

            // Clean up
            stream.getTracks().forEach(track => track.stop());
            if (this.audioContext) {
                this.audioContext.close();
            }
        };

        this.mediaRecorder.start();
        
        // Safety timeout (max 10 seconds)
        setTimeout(() => {
            if (this.isRecording) {
                console.log("⏱️ Max duration reached, stopping...");
                this.stopRecording();
            }
        }, 10000);
    },

    stopRecording() {
        if (this.mediaRecorder && this.isRecording) {
            this.mediaRecorder.stop();
            this.isRecording = false;
            this.voiced = false;
            
            if (this.javascriptNode) {
                this.javascriptNode.disconnect();
            }
            if (this.analyser) {
                this.analyser.disconnect();
            }
            if (this.microphone) {
                this.microphone.disconnect();
            }
        }
    },

    updateVolumeIndicator(level) {
        // Visual feedback for user
        const indicator = document.getElementById("recording-indicator");
        if (!indicator) return;

        if (level > this.VOICE_THRESHOLD) {
            // Voice detected - show green
            indicator.style.backgroundColor = '#10b981';
            indicator.classList.remove('animate-pulse');
        } else {
            // Silence - show red pulsing
            indicator.style.backgroundColor = '#ef4444';
            indicator.classList.add('animate-pulse');
        }
    },

    async processAudio(audioBlob) {
        const status = document.getElementById("voice-status");
        status.textContent = "⚙️ Processing audio...";

        const mimeType = audioBlob.type || "audio/webm";
        let format = "webm";
        if (mimeType.includes("webm")) format = "webm";
        else if (mimeType.includes("ogg")) format = "ogg";
        else if (mimeType.includes("mp4")) format = "mp4";
        else if (mimeType.includes("wav")) format = "wav";

        console.log("Sending format:", format, "Size:", audioBlob.size);

        const base64Audio = await this.blobToBase64(audioBlob);

        try {
            // Extract MFCC from backend
            const response = await fetch(
                "http://127.0.0.1:8000/api/enroll/extract-mfcc", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    audio_data: base64Audio,
                    audio_format: format,
                    username: typeof authUsername !== 'undefined'
                              ? authUsername
                              : typeof currentUsername !== 'undefined'
                              ? currentUsername
                              : ""
                })
            });

            const result = await response.json();
            console.log("MFCC result:", result);

            if (!result.success) {
                status.textContent = "❌ " + (result.detail || "Try again");
                status.className = "text-center text-sm mb-4 text-red-400";
                
                // Reset button for retry
                const btn = document.getElementById("record-btn");
                btn.disabled = false;
                btn.textContent = "🎤 Try Again";
                return;
            }

            const mfccFeatures = result.mfcc_features;

            // Detect which page and call appropriate function
            if (typeof onVoiceAuthComplete === 'function') {
                // LOGIN PAGE
                status.textContent = "⏳ Verifying voice...";
                onVoiceAuthComplete(mfccFeatures);

            } else if (typeof onVoiceRecorded === 'function') {
                // ENROLLMENT PAGE
                status.textContent = `✅ Recording ${this.currentAttempt} processed!`;
                status.className = "text-center text-sm mb-4 text-green-400";
                onVoiceRecorded(mfccFeatures);
                this.currentAttempt++;

                const btn = document.getElementById("record-btn");
                if (this.currentAttempt <= this.maxAttempts) {
                    btn.textContent = 
                        `🎤 Record Again (${this.currentAttempt}/${this.maxAttempts})`;
                    btn.disabled = false;
                    btn.onclick = startRecording;
                } else {
                    btn.disabled = true;
                    btn.textContent = "✅ All recordings done";
                    btn.classList.replace("bg-red-600", "bg-gray-600");
                }
            }

        } catch (err) {
            console.error("Audio processing error:", err);
            status.textContent = "❌ Could not connect to server.";
            status.className = "text-center text-sm mb-4 text-red-400";
        }
    },

    blobToBase64(blob) {
        return new Promise((resolve, reject) => {
            const reader = new FileReader();
            reader.onloadend = () => {
                const base64 = reader.result.split(',')[1];
                resolve(base64);
            };
            reader.onerror = reject;
            reader.readAsDataURL(blob);
        });
    },

    reset() {
        this.audioChunks = [];
        this.isRecording = false;
        this.currentAttempt = 1;
        this.mediaRecorder = null;
        this.voiced = false;
        this.silenceStart = null;
        
        if (this.audioContext) {
            this.audioContext.close();
            this.audioContext = null;
        }
    }
};

// Global function called by HTML button
function startRecording() {
    const btn = document.getElementById("record-btn");
    const indicator = document.getElementById("recording-indicator");
    const status = document.getElementById("voice-status");

    btn.disabled = true;
    btn.textContent = "🎤 Listening for voice...";
    btn.classList.replace("bg-red-600", "bg-yellow-600");
    indicator.classList.remove("hidden");
    indicator.style.backgroundColor = '#ef4444';
    status.textContent = "🎤 Speak now - recording will start when voice detected";
    status.className = "text-center text-sm mb-4 text-yellow-400";

    SpeechCapture.startRecording();
}