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

        // ✅ Detect format from actual mime type
        const mimeType = audioBlob.type || "audio/webm";
        let format = "webm"; // default

        if (mimeType.includes("webm")) format = "webm";
        else if (mimeType.includes("ogg")) format = "ogg";
        else if (mimeType.includes("mp4")) format = "mp4";
        else if (mimeType.includes("wav")) format = "wav";

        console.log("Sending format:", format);

        const base64Audio = await this.blobToBase64(audioBlob);

        try {
            const response = await fetch("http://127.0.0.1:8000/api/enroll/extract-mfcc", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ 
                    audio_data: base64Audio,
                    audio_format: format,       // ✅ send format to backend
                    username: currentUsername 
                })
            });

            const result = await response.json();
            console.log("MFCC result:", result);

            if (result.success) {
                status.textContent = `✅ Recording processed!`;

                // Check which page we're on
                if (typeof onVoiceAuthComplete === 'function') {
                    // Login page — verify voice
                    onVoiceAuthComplete(result.mfcc_features);
                } else if (typeof onVoiceRecorded === 'function') {
                    // Enrollment page — save voice
                    onVoiceRecorded(result.mfcc_features);
                    this.currentAttempt++;

                    const btn = document.getElementById("record-btn");
                    if (this.currentAttempt <= this.maxAttempts) {
                        btn.textContent = `🎤 Record Again (${this.currentAttempt}/${this.maxAttempts})`;
                        btn.disabled = false;
                        btn.onclick = startRecording;
                    } else {
                        btn.disabled = true;
                        btn.textContent = "✅ All recordings done";
                        btn.classList.replace("bg-red-600", "bg-gray-600");
                    }
                }
                this.currentAttempt++;

                const btn = document.getElementById("record-btn");
                if (this.currentAttempt <= this.maxAttempts) {
                    btn.textContent = `🎤 Record Again (${this.currentAttempt}/${this.maxAttempts})`;
                    btn.disabled = false;
                    btn.onclick = startRecording;
                } else {
                    btn.disabled = true;
                    btn.textContent = "✅ All recordings done";
                    btn.classList.replace("bg-red-600", "bg-gray-600");
                }
            } else {
                status.textContent = "❌ Failed: " + (result.detail || "Try again");
                console.error("Server error:", result.detail);
            }

        } catch (err) {
            console.error("Fetch error:", err);
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