// frontend/js/auth_flow.js
// Controls the progressive authentication flow (Design 3)

let authUsername = "";
let failedAttempts = 0;
const MAX_ATTEMPTS = 3;

// ── Step 0: Start Login ───────────────────────────────────
function startLogin() {
    const username = document.getElementById("username-input").value.trim();
    if (!username) {
        alert("Please enter your username.");
        return;
    }

    authUsername = username;

    // Show step indicator and keystroke section
    document.getElementById("username-section").classList.add("hidden");
    document.getElementById("step-indicator").classList.remove("hidden");
    document.getElementById("keystroke-section").classList.remove("hidden");
    document.getElementById("attempts-indicator").classList.remove("hidden");

    // Activate step 1 dot
    document.getElementById("dot-keystroke")
        .classList.replace("bg-gray-700", "bg-purple-600");

    // Attach keystroke capture
    KeystrokeCapture.attach("keystroke-input");

    document.getElementById("keystroke-status").textContent =
        "Start typing when ready";
}

// ── Step 1: Keystroke Auth ────────────────────────────────
async function submitKeystrokeAuth() {
    const input = document.getElementById("keystroke-input");
    const status = document.getElementById("keystroke-status");

    // Validate phrase
    if (!KeystrokeCapture.validatePhrase(input.value)) {
        status.textContent = "❌ Phrase doesn't match. Type exactly as shown.";
        status.className = "text-center text-sm mb-4 text-red-400";
        input.value = "";
        KeystrokeCapture.reset();
        KeystrokeCapture.attach("keystroke-input");
        return;
    }

    const features = KeystrokeCapture.extractFeatures();
    if (!features || features.totalKeys < 5) {
        status.textContent = "❌ Not enough data. Try again.";
        return;
    }

    status.textContent = "⏳ Verifying keystroke pattern...";
    status.className = "text-center text-sm mb-4 text-yellow-400";

    try {
        const result = await Api.verifyKeystroke(authUsername, features);

        if (result.authenticated) {
            // ✅ Keystroke passed
            showSuccess("Keystroke Dynamics", result.confidence);
        } else {
            // ❌ Keystroke failed — move to voice
            recordFailedAttempt();
            status.textContent = `❌ Keystroke failed (confidence: ${(result.confidence * 100).toFixed(1)}%)`;
            status.className = "text-center text-sm mb-4 text-red-400";

            setTimeout(() => moveToVoiceAuth(), 1500);
        }

    } catch (err) {
        console.error(err);
        status.textContent = "❌ Server error. Try again.";
        status.className = "text-center text-sm mb-4 text-red-400";
    }
}

// ── Step 2: Voice Auth ────────────────────────────────────
function moveToVoiceAuth() {
    document.getElementById("keystroke-section").classList.add("hidden");
    document.getElementById("voice-section").classList.remove("hidden");

    // Update step indicator
    document.getElementById("dot-voice")
        .classList.replace("bg-gray-700", "bg-purple-600");
    document.getElementById("line-1").style.width = "100%";
}

function startVoiceAuth() {
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

// Called from speech.js after recording is processed
async function onVoiceAuthComplete(mfccFeatures) {
    const status = document.getElementById("voice-status");
    status.textContent = "⏳ Verifying voice pattern...";

    try {
        const result = await Api.verifyVoice(authUsername, mfccFeatures);

        if (result.authenticated) {
            showSuccess("Speech Biometrics", result.confidence);
        } else {
            recordFailedAttempt();
            status.textContent =
                `❌ Voice failed (confidence: ${(result.confidence * 100).toFixed(1)}%)`;
            status.className = "text-center text-sm mb-4 text-red-400";
            setTimeout(() => moveToSecurityAuth(), 1500);
        }

    } catch (err) {
        console.error(err);
        status.textContent = "❌ Server error. Try again.";
    }
}

// ── Step 3: Security Question Auth ───────────────────────
async function moveToSecurityAuth() {
    document.getElementById("voice-section").classList.add("hidden");
    document.getElementById("security-section").classList.remove("hidden");

    // Update step indicator
    document.getElementById("dot-security")
        .classList.replace("bg-gray-700", "bg-purple-600");
    document.getElementById("line-2").style.width = "100%";

    // Fetch the user's security question from backend
    try {
        const result = await Api.getSecurityQuestion(authUsername);
        if (result.question) {
            document.getElementById("security-question-text").textContent =
                result.question;
        } else {
            document.getElementById("security-question-text").textContent =
                "Security question not found.";
        }
    } catch (err) {
        console.error(err);
    }
}

async function submitSecurityAuth() {
    const answer = document.getElementById("security-answer-input").value.trim();
    const status = document.getElementById("security-status");

    if (!answer) {
        status.textContent = "Please enter your answer.";
        return;
    }

    status.textContent = "⏳ Verifying answer...";

    try {
        const result = await Api.verifySecurityQuestion(authUsername, answer);

        if (result.authenticated) {
            // ✅ Security question passed
            showSuccess("Security Question", result.confidence);
        } else {
            // ❌ All methods failed
            recordFailedAttempt();
            showDenied();
        }

    } catch (err) {
        console.error(err);
        status.textContent = "❌ Server error. Try again.";
    }
}

// ── Helpers ───────────────────────────────────────────────
function recordFailedAttempt() {
    failedAttempts++;
    const dot = document.getElementById(`fail-${failedAttempts}`);
    if (dot) dot.classList.replace("bg-gray-700", "bg-red-500");
}

function showSuccess(method, confidence) {
    // Hide all sections
    hideAllSections();

    document.getElementById("success-section").classList.remove("hidden");
    document.getElementById("success-method").textContent =
        `Verified via: ${method}`;
    document.getElementById("success-confidence").textContent =
        `Confidence: ${(confidence * 100).toFixed(1)}%`;
}

function showDenied() {
    hideAllSections();
    document.getElementById("denied-section").classList.remove("hidden");
}

function hideAllSections() {
    const sections = [
        "username-section", "keystroke-section",
        "voice-section", "security-section",
        "success-section", "denied-section"
    ];
    sections.forEach(id => {
        document.getElementById(id).classList.add("hidden");
    });
}

function resetLogin() {
    failedAttempts = 0;
    authUsername = "";
    KeystrokeCapture.reset();
    SpeechCapture.reset();

    // Reset fail dots
    [1, 2, 3].forEach(i => {
        const dot = document.getElementById(`fail-${i}`);
        if (dot) dot.classList.replace("bg-red-500", "bg-gray-700");
    });

    // Reset step dots
    ["dot-keystroke", "dot-voice", "dot-security"].forEach(id => {
        document.getElementById(id)
            .classList.replace("bg-purple-600", "bg-gray-700");
    });

    // Reset lines
    document.getElementById("line-1").style.width = "0%";
    document.getElementById("line-2").style.width = "0%";

    // Show username section
    hideAllSections();
    document.getElementById("username-section").classList.remove("hidden");
    document.getElementById("step-indicator").classList.add("hidden");
    document.getElementById("attempts-indicator").classList.add("hidden");
    document.getElementById("username-input").value = "";
}