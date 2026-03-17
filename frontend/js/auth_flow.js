// frontend/js/auth_flow.js

let authUsername   = "";
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

    document.getElementById("username-section").classList.add("hidden");
    document.getElementById("step-indicator").classList.remove("hidden");
    document.getElementById("password-section").classList.remove("hidden");  // ← UPDATED
    document.getElementById("attempts-indicator").classList.remove("hidden");

    document.getElementById("dot-password")
        .classList.replace("bg-gray-700", "bg-purple-600");  // ← UPDATED
}

// ── Step 1: Password Auth ← ADDED ────────────────────────
async function submitPasswordAuth() {
    const password = document.getElementById("password-input").value;
    const status   = document.getElementById("password-status");

    if (!password) {
        status.textContent = "Please enter your password.";
        status.className   = "text-center text-sm mb-4 text-red-400";
        return;
    }

    status.textContent = "⏳ Verifying password...";
    status.className   = "text-center text-sm mb-4 text-yellow-400";

    try {
        const result = await Api.verifyPassword(authUsername, password);

        if (result.authenticated) {
            status.textContent = "✅ Password verified!";
            status.className   = "text-center text-sm mb-4 text-green-400";
            setTimeout(() => moveToKeystrokeAuth(), 800);
        } else {
            recordFailedAttempt();
            status.textContent = "❌ Incorrect password.";
            status.className   = "text-center text-sm mb-4 text-red-400";

            if (failedAttempts >= MAX_ATTEMPTS) {
                setTimeout(() => moveToSecurityAuth(), 1500);
            }
        }
    } catch (err) {
        console.error(err);
        status.textContent = "❌ Server error. Try again.";
        status.className   = "text-center text-sm mb-4 text-red-400";
    }
}

// ── Step 2: Keystroke Auth ────────────────────────────────
function moveToKeystrokeAuth() {  // ← ADDED
    document.getElementById("password-section").classList.add("hidden");
    document.getElementById("keystroke-section").classList.remove("hidden");

    document.getElementById("dot-keystroke")
        .classList.replace("bg-gray-700", "bg-purple-600");
    document.getElementById("line-1").style.width = "100%";

    KeystrokeCapture.attach("keystroke-input");
    document.getElementById("keystroke-status").textContent = "Start typing when ready";
}

async function submitKeystrokeAuth() {
    const input  = document.getElementById("keystroke-input");
    const status = document.getElementById("keystroke-status");

    if (!KeystrokeCapture.validatePhrase(input.value)) {
        status.textContent = "❌ Phrase doesn't match. Type exactly as shown.";
        status.className = "text-center text-sm mb-4 text-red-400";
        input.value = "";
        KeystrokeCapture.reset();
        KeystrokeCapture.attach("keystroke-input");
        return;
    }

    const features = KeystrokeCapture.extractFeatures();
    if (!features) {
        status.textContent = "❌ Not enough data. Try again.";
        status.className = "text-center text-sm mb-4 text-red-400";
        return;
    }

    status.textContent = "⏳ Verifying keystroke pattern...";
    status.className = "text-center text-sm mb-4 text-yellow-400";

    try {
        const result = await Api.verifyKeystroke(authUsername, features);

        if (result.authenticated) {
            showSuccess("Keystroke Dynamics", result.confidence);
        } else {
            recordFailedAttempt();
            status.textContent =
                `❌ Keystroke failed (confidence: ${(result.confidence * 100).toFixed(1)}%)`;
            status.className = "text-center text-sm mb-4 text-red-400";
            setTimeout(() => moveToVoiceAuth(), 1500);
        }

    } catch (err) {
        console.error(err);
        status.textContent = "❌ Server error. Try again.";
        status.className = "text-center text-sm mb-4 text-red-400";
    }
}

// ── Step 3: Voice Auth ────────────────────────────────────
function moveToVoiceAuth() {
    document.getElementById("keystroke-section").classList.add("hidden");
    document.getElementById("voice-section").classList.remove("hidden");

    document.getElementById("dot-voice")
        .classList.replace("bg-gray-700", "bg-purple-600");
    document.getElementById("line-2").style.width = "100%";
}

function startVoiceAuth() {
    const btn       = document.getElementById("record-btn");
    const indicator = document.getElementById("recording-indicator");
    const status    = document.getElementById("voice-status");

    if (SpeechCapture.isRecording) {
        SpeechCapture.stopRecording();
        return;
    }

    // Use the same full VAD pipeline as enrollment — noise floor estimate,
    // speech-band check, auto-stop on silence. The old 4-second timer bypass
    // skipped all quality checks, causing low-quality recordings to reach the model.
    btn.disabled    = true;
    btn.textContent = "🎤 Measuring background…";
    btn.classList.replace("bg-red-600", "bg-yellow-600");
    indicator.classList.remove("hidden");
    indicator.style.backgroundColor = "#f59e0b";
    status.textContent = "🔇 Measuring background noise (stay quiet)…";
    status.className   = "text-center text-sm mb-4 text-yellow-400";

    SpeechCapture.reset();
    SpeechCapture.startRecording().then(ok => {
        if (ok) {
            btn.textContent = "🎤 Listening — speak the phrase…";
            btn.classList.replace("bg-yellow-600", "bg-red-600");
        } else {
            // startRecording already set an error status; just re-enable the button
            btn.disabled    = false;
            btn.textContent = "🎤 Try Again";
            btn.classList.replace("bg-yellow-600", "bg-red-600");
            indicator.classList.add("hidden");
        }
    });
}

async function onVoiceAuthComplete(fullFeatureDict) {
    const status = document.getElementById("voice-status");
    const btn    = document.getElementById("record-btn");
    status.textContent = "⏳ Verifying voice pattern...";
    status.className   = "text-center text-sm mb-4 text-yellow-400";
    if (btn) btn.disabled = true;

    try {
        const result = await Api.verifyVoice(authUsername, fullFeatureDict);

        if (result.authenticated) {
            showSuccess("Speech Biometrics", result.confidence);
        } else {
            recordFailedAttempt();
            // Show fused_score (0–100 scale) if available, otherwise fall back to confidence
            const pct = result.fused_score != null
                ? result.fused_score.toFixed(1)
                : (result.confidence * 100).toFixed(1);
            status.textContent =
                `❌ Voice not recognised (score: ${pct}%). Speak clearly and try again.`;
            status.className = "text-center text-sm mb-4 text-red-400";

            if (btn) {
                btn.disabled    = false;
                btn.textContent = "🎤 Try Again";
            }

            // Only advance to security question after 2 voice failures, not 1
            const voiceFailures = parseInt(btn?.dataset.voiceFails || "0") + 1;
            if (btn) btn.dataset.voiceFails = voiceFailures;
            if (voiceFailures >= 2) {
                setTimeout(() => moveToSecurityAuth(), 1500);
            }
        }

    } catch (err) {
        console.error(err);
        status.textContent = "❌ Server error. Try again.";
        status.className   = "text-center text-sm mb-4 text-red-400";
        if (btn) { btn.disabled = false; btn.textContent = "🎤 Try Again"; }
    }
}

// ── Step 4: Security Question Auth ───────────────────────
async function moveToSecurityAuth() {
    document.getElementById("voice-section").classList.add("hidden");
    document.getElementById("security-section").classList.remove("hidden");

    document.getElementById("dot-security")
        .classList.replace("bg-gray-700", "bg-purple-600");
    document.getElementById("line-3").style.width = "100%";  // ← UPDATED

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
    status.className = "text-center text-sm mb-4 text-yellow-400";

    try {
        const result = await Api.verifySecurityQuestion(authUsername, answer);

        if (result.authenticated) {
            showSuccess("Security Question", result.confidence);
        } else {
            recordFailedAttempt();
            showDenied();
        }

    } catch (err) {
        console.error(err);
        status.textContent = "❌ Server error. Try again.";
        status.className = "text-center text-sm mb-4 text-red-400";
    }
}

// ── Helpers ───────────────────────────────────────────────
function recordFailedAttempt() {
    failedAttempts++;
    const dot = document.getElementById(`fail-${failedAttempts}`);
    if (dot) dot.classList.replace("bg-gray-700", "bg-red-500");
}

function showSuccess(method, confidence) {
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
        "username-section", "password-section",   // ← UPDATED
        "keystroke-section", "voice-section",
        "security-section", "success-section", "denied-section"
    ];
    sections.forEach(id => {
        document.getElementById(id).classList.add("hidden");
    });
}

function resetLogin() {
    failedAttempts = 0;
    authUsername   = "";
    KeystrokeCapture.reset();
    SpeechCapture.reset();
    const recordBtn = document.getElementById("record-btn");
    if (recordBtn) recordBtn.dataset.voiceFails = "0";

    [1, 2, 3].forEach(i => {
        const dot = document.getElementById(`fail-${i}`);
        if (dot) dot.classList.replace("bg-red-500", "bg-gray-700");
    });

    ["dot-password", "dot-keystroke",               // ← UPDATED
     "dot-voice", "dot-security"].forEach(id => {
        const el = document.getElementById(id);
        if (el) el.classList.replace("bg-purple-600", "bg-gray-700");
    });

    document.getElementById("line-1").style.width = "0%";
    document.getElementById("line-2").style.width = "0%";
    document.getElementById("line-3").style.width = "0%";  // ← ADDED

    hideAllSections();
    document.getElementById("username-section").classList.remove("hidden");
    document.getElementById("step-indicator").classList.add("hidden");
    document.getElementById("attempts-indicator").classList.add("hidden");
    document.getElementById("username-input").value = "";
}