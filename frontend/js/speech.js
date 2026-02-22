// frontend/js/speech.js

const SpeechCapture = {
    mediaRecorder: null,
    audioChunks: [],
    isRecording: false,
    currentAttempt: 1,
    maxAttempts: 3,

    async startRecording() {
        try {
            const stream = await navigator.mediaDevices.getUserMedia({ 
                audio: {
                    channelCount: 1,
                    echoCancellation: true,
                    noiseSuppression: true
                }
            });

            this.audioChunks = [];
            this.mediaRecorder = new MediaRecorder(stream);
            this.isRecording = true;

            this.mediaRecorder.ondataavailable = (event) => {
                if (event.data.size > 0) {
                    this.audioChunks.push(event.data);
                }
            };

            this.mediaRecorder.onstop = () => {
                // ✅ Use actual mime type browser recorded with
                const mimeType = this.mediaRecorder.mimeType || 'audio/webm';
                const audioBlob = new Blob(this.audioChunks, { type: mimeType });

                console.log("Recorded mime type:", mimeType);
                console.log("Audio blob size:", audioBlob.size, "bytes");

                this.processAudio(audioBlob);
                stream.getTracks().forEach(track => track.stop());
            };

            this.mediaRecorder.start();

            // Auto-stop after 4 seconds
            setTimeout(() => {
                if (this.isRecording) {
                    this.stopRecording();
                }
            }, 4000);

            return true;

        } catch (err) {
            console.error("Microphone error:", err);
            document.getElementById("voice-status").textContent = 
                "❌ Microphone access denied. Please allow microphone and try again.";
            return false;
        }
    },

    stopRecording() {
        if (this.mediaRecorder && this.isRecording) {
            this.mediaRecorder.stop();
            this.isRecording = false;
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

    console.log("Audio format:", format, "Size:", audioBlob.size);

    const base64Audio = await this.blobToBase64(audioBlob);

    try {
        // Step 1 — Always extract MFCC first via enroll endpoint
        const mfccResponse = await fetch(
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

        const mfccResult = await mfccResponse.json();
        console.log("MFCC result:", mfccResult);

        if (!mfccResult.success) {
            status.textContent = "❌ Failed: " + (mfccResult.detail || "Try again");
            return;
        }

        const mfccFeatures = mfccResult.mfcc_features;

        // Step 2 — Detect which page we're on and call correct function
        if (typeof onVoiceAuthComplete === 'function') {
            // ── LOGIN PAGE ──
            // MFCC extracted, now verify against stored template
            status.textContent = "⏳ Verifying voice...";
            onVoiceAuthComplete(mfccFeatures);

        } else if (typeof onVoiceRecorded === 'function') {
            // ── ENROLLMENT PAGE ──
            // MFCC extracted, save it
            status.textContent = `✅ Recording ${this.currentAttempt} processed!`;
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
    }
};

// Global function called by the HTML button
function startRecording() {
    const btn = document.getElementById("record-btn");
    const indicator = document.getElementById("recording-indicator");
    const status = document.getElementById("voice-status");

    if (SpeechCapture.isRecording) {
        SpeechCapture.stopRecording();
        btn.textContent = "🎤 Start Recording";
        indicator.classList.add("hidden");
        return;
    }

    btn.textContent = "⏹ Stop Recording";
    btn.classList.replace("bg-red-600", "bg-red-800");
    indicator.classList.remove("hidden");
    status.textContent = "🔴 Recording... (4 seconds)";

    SpeechCapture.startRecording().then(started => {
        if (started) {
            setTimeout(() => {
                btn.textContent = "🎤 Start Recording";
                btn.classList.replace("bg-red-800", "bg-red-600");
                indicator.classList.add("hidden");
            }, 4200);
        }
    });
}