// frontend/js/auth_flow.js

let authUsername   = "";
let failedAttempts = 0;
const MAX_ATTEMPTS = 3;

// ── Fusion & Decision Module (Figure 9) ──────────────────
// Scores stored when each modality fails its individual threshold.
// If both fail, fusion runs before escalating to security question.
// Weights: keystroke 0.45, voice 0.55 | Threshold: 0.50
const FUSION_WEIGHT_KS    = 0.45;
const FUSION_WEIGHT_VOICE = 0.55;
const FUSION_THRESHOLD    = 0.50;

let _ksScore    = null;
let _voiceScore = null;


// ── Step 0: Login (email + password on one screen) ───────────────────────
async function submitLogin() {
    const rawEmail = document.getElementById("username-input").value.trim();
    const password = document.getElementById("password-input").value;
    const status   = document.getElementById("login-status");

    if (!rawEmail) {
        status.textContent = "Please enter your email address.";
        status.className   = "text-center text-sm mb-4 text-red-400";
        return;
    }
    if (!password) {
        status.textContent = "Please enter your password.";
        status.className   = "text-center text-sm mb-4 text-red-400";
        return;
    }

    const email = rawEmail.toLowerCase();

    status.textContent = "⏳ Verifying credentials...";
    status.className   = "text-center text-sm mb-4 text-yellow-400";

    try {
        const result = await Api.verifyPassword(email, password);

        if (result.authenticated) {
            authUsername = email;
            status.textContent = "✅ Credentials verified!";
            status.className   = "text-center text-sm mb-4 text-green-400";

            // Fetch the user's unique phrase before moving to keystroke step
            try {
                const phraseResult = await Api.getPhrase(email);
                const phrase = phraseResult.phrase || "";
                if (phrase) {
                    KeystrokeCapture.setPhrase(phrase);
                    document.querySelectorAll(".phrase-display").forEach(el => {
                        el.textContent = phrase;
                    });
                }
            } catch (e) {
                console.warn("Could not fetch phrase:", e);
            }

            setTimeout(() => moveToKeystrokeAuth(), 800);
        } else {
            recordFailedAttempt();
            status.textContent = "❌ Incorrect email or password.";
            status.className   = "text-center text-sm mb-4 text-red-400";
        }
    } catch (err) {
        console.error(err);
        status.textContent = "❌ Server error. Try again.";
        status.className   = "text-center text-sm mb-4 text-red-400";
    }
}


// ── Step 1: Keystroke Auth ────────────────────────────────
function moveToKeystrokeAuth() {
    // Hide all sections cleanly before showing keystroke
    // This prevents security section bleeding into keystroke during re-auth
    hideAllSections();
    document.getElementById("step-indicator").classList.remove("hidden");
    document.getElementById("keystroke-section").classList.remove("hidden");
    document.getElementById("attempts-indicator").classList.remove("hidden");

    document.getElementById("dot-keystroke")
        .classList.replace("bg-gray-700", "bg-purple-600");

    // Clear any leftover text from previous attempt
    const input = document.getElementById("keystroke-input");
    if (input) input.value = "";

    KeystrokeCapture.reset();
    KeystrokeCapture.attach("keystroke-input");
    document.getElementById("keystroke-status").textContent = "Start typing when ready";
}

async function submitKeystrokeAuth() {
    const input  = document.getElementById("keystroke-input");
    const status = document.getElementById("keystroke-status");

    if (!KeystrokeCapture.validatePhrase(input.value)) {
        status.textContent = "❌ Phrase doesn't match. Type exactly as shown.";
        status.className   = "text-center text-sm mb-4 text-red-400";
        input.value = "";
        KeystrokeCapture.reset();
        KeystrokeCapture.attach("keystroke-input");
        return;
    }

    const features = KeystrokeCapture.extractFeatures();
    if (!features) {
        status.textContent = "❌ Not enough data. Try again.";
        status.className   = "text-center text-sm mb-4 text-red-400";
        return;
    }

    status.textContent = "⏳ Verifying keystroke pattern...";
    status.className   = "text-center text-sm mb-4 text-yellow-400";

    try {
        const result = await Api.verifyKeystroke(authUsername, features);

        if (result.authenticated) {
            // Keystroke passed individually → grant immediately
            showSuccess("Keystroke Dynamics", result.confidence);
        } else {
            // Store score for fusion, move to voice
            _ksScore = typeof result.confidence === "number" ? result.confidence : 0.0;
            recordFailedAttempt();
            status.textContent =
                `❌ Keystroke not matched (${(_ksScore * 100).toFixed(1)}%) — trying voice…`;
            status.className = "text-center text-sm mb-4 text-red-400";
            setTimeout(() => moveToVoiceAuth(), 1500);
        }

    } catch (err) {
        console.error(err);
        status.textContent = "❌ Server error. Try again.";
        status.className   = "text-center text-sm mb-4 text-red-400";
    }
}


// ── Step 2: Voice Auth ────────────────────────────────────
function moveToVoiceAuth() {
    hideAllSections();
    document.getElementById("step-indicator").classList.remove("hidden");
    document.getElementById("voice-section").classList.remove("hidden");
    document.getElementById("attempts-indicator").classList.remove("hidden");

    document.getElementById("dot-voice")
        .classList.replace("bg-gray-700", "bg-purple-600");
    document.getElementById("line-1").style.width = "100%";
}

function startVoiceAuth() {
    const btn       = document.getElementById("record-btn");
    const tryAgain  = document.getElementById("try-again-btn");
    const indicator = document.getElementById("recording-indicator");
    const status    = document.getElementById("voice-status");

    if (SpeechCapture.isRecording) {
        SpeechCapture.stopRecording();
        return;
    }

    btn.disabled    = true;
    btn.textContent = "🎤 Measuring background…";
    btn.classList.replace("bg-red-600", "bg-yellow-600");
    if (tryAgain) tryAgain.disabled = true;
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
            btn.disabled    = false;
            btn.textContent = "🎤 Try Again";
            btn.classList.replace("bg-yellow-600", "bg-red-600");
            if (tryAgain) tryAgain.disabled = false;
            indicator.classList.add("hidden");
        }
    });
}

// Resets the voice step so the user can try a fresh recording
function resetVoiceAuth() {
    if (SpeechCapture.isRecording) SpeechCapture.stopRecording();
    SpeechCapture.reset();

    const btn      = document.getElementById("record-btn");
    const tryAgain = document.getElementById("try-again-btn");
    const indicator = document.getElementById("recording-indicator");
    const status    = document.getElementById("voice-status");
    const diagBar   = document.getElementById("diag-bar");
    const diagText  = document.getElementById("diag-text");

    if (btn) {
        btn.disabled    = false;
        btn.textContent = "🎤 Start Recording";
        btn.classList.replace("bg-yellow-600", "bg-red-600");
    }
    if (tryAgain)  tryAgain.disabled = false;
    if (indicator) indicator.classList.add("hidden");
    if (diagBar)   { diagBar.style.width = "0%"; diagBar.style.background = "#ef4444"; }
    if (diagText)  diagText.textContent = "";
    if (status) {
        status.textContent = "Click record when ready";
        status.className   = "text-center text-sm mb-4 text-gray-500";
    }
}

async function onVoiceAuthComplete(fullFeatureDict) {
    const status = document.getElementById("voice-status");
    const btn    = document.getElementById("record-btn");
    status.textContent = "⏳ Verifying voice pattern...";
    status.className   = "text-center text-sm mb-4 text-yellow-400";
    if (btn) btn.disabled = true;
    const tryAgain = document.getElementById("try-again-btn");
    if (tryAgain) tryAgain.disabled = true;

    try {
        const result = await Api.verifyVoice(authUsername, fullFeatureDict);

        const voiceScore = result.fused_score != null
            ? result.fused_score / 100.0
            : (typeof result.confidence === "number" ? result.confidence : 0.0);

        if (result.authenticated) {
            // Voice passed individually → grant immediately
            showSuccess("Speech Biometrics", result.confidence);
        } else {
            _voiceScore = voiceScore;
            recordFailedAttempt();

            // ── Fusion & Decision Module ──────────────────────
            const ksScore    = _ksScore !== null ? _ksScore : 0.0;
            const fusedScore = (FUSION_WEIGHT_KS * ksScore) + (FUSION_WEIGHT_VOICE * _voiceScore);

            console.log(
                `[Fusion & Decision Module] ` +
                `keystroke=${(ksScore * 100).toFixed(1)}%  ` +
                `voice=${(_voiceScore * 100).toFixed(1)}%  ` +
                `fused=${(fusedScore * 100).toFixed(1)}%  ` +
                `threshold=${(FUSION_THRESHOLD * 100).toFixed(0)}%`
            );

            if (fusedScore >= FUSION_THRESHOLD) {
                // Fusion passed → grant access
                showSuccess("Fusion & Decision Module", fusedScore);
            } else {
                // Fusion failed → security question
                status.textContent =
                    `❌ Voice not matched (${(voiceScore * 100).toFixed(1)}%) — ` +
                    `fused: ${(fusedScore * 100).toFixed(1)}% — proceeding to security question…`;
                status.className = "text-center text-sm mb-4 text-red-400";
                setTimeout(() => moveToSecurityAuth(), 800);
            }
        }

    } catch (err) {
        console.error(err);
        status.textContent = "❌ Server error. Try again.";
        status.className   = "text-center text-sm mb-4 text-red-400";
        if (btn) { btn.disabled = false; btn.textContent = "🎤 Try Again"; }
        if (tryAgain) tryAgain.disabled = false;
    }
}


// ── Step 3: Security Question ─────────────────────────────
async function moveToSecurityAuth() {
    // Stop any active voice recording before transitioning
    if (SpeechCapture.isRecording) SpeechCapture.stopRecording();
    SpeechCapture.reset();

    // Hide ALL sections cleanly — prevents voice UI bleeding into security section
    hideAllSections();
    document.getElementById("security-section").classList.remove("hidden");

    document.getElementById("dot-security")
        .classList.replace("bg-gray-700", "bg-purple-600");
    document.getElementById("line-2").style.width = "100%";

    try {
        const result = await Api.getSecurityQuestion(authUsername);
        document.getElementById("security-question-text").textContent =
            result.question || "Security question not found.";
    } catch (err) {
        console.error(err);
    }
}

async function submitSecurityAuth() {
    const answer = document.getElementById("security-answer-input").value.trim();
    const status = document.getElementById("security-status");

    if (!answer) {
        status.textContent = "Please enter your answer.";
        status.className   = "text-center text-sm mb-4 text-red-400";
        return;
    }

    status.textContent = "⏳ Verifying answer...";
    status.className   = "text-center text-sm mb-4 text-yellow-400";

    try {
        const result = await Api.verifySecurityQuestion(authUsername, answer);

        if (result.authenticated) {
            // Correct → grant access directly (security question IS the fallback auth)
            status.textContent = "✅ Correct! Access granted.";
            status.className   = "text-center text-sm mb-4 text-green-400";
            setTimeout(() => {
                showSuccess("Security Question", 1.0);
            }, 800);
        } else {
            // Wrong → flagged and rejected
            recordFailedAttempt();
            showDenied();
        }

    } catch (err) {
        console.error(err);
        status.textContent = "❌ Server error. Try again.";
        status.className   = "text-center text-sm mb-4 text-red-400";
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
    [
        "login-section", "keystroke-section", "voice-section",
        "security-section", "success-section", "denied-section"
    ].forEach(id => document.getElementById(id)?.classList.add("hidden"));
}

function resetLogin() {
    failedAttempts = 0;
    authUsername   = "";
    _ksScore       = null;
    _voiceScore    = null;
    KeystrokeCapture.reset();
    SpeechCapture.reset();

    const recordBtn = document.getElementById("record-btn");
    if (recordBtn) recordBtn.dataset.voiceFails = "0";

    [1, 2, 3].forEach(i => {
        const dot = document.getElementById(`fail-${i}`);
        if (dot) dot.classList.replace("bg-red-500", "bg-gray-700");
    });

    ["dot-keystroke", "dot-voice", "dot-security"].forEach(id => {
        const el = document.getElementById(id);
        if (el) el.classList.replace("bg-purple-600", "bg-gray-700");
    });

    document.getElementById("line-1").style.width = "0%";
    document.getElementById("line-2").style.width = "0%";

    hideAllSections();
    document.getElementById("login-section").classList.remove("hidden");
    document.getElementById("step-indicator").classList.add("hidden");
    document.getElementById("attempts-indicator").classList.add("hidden");
    document.getElementById("username-input").value = "";
    document.getElementById("password-input").value = "";
    document.getElementById("login-status").textContent = "";
    document.querySelectorAll(".phrase-display").forEach(el => {
        el.textContent = "Loading your phrase…";
    });
    KeystrokeCapture.setPhrase("");
}