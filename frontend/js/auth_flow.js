// frontend/js/auth_flow.js

let authUsername   = "";
let failedAttempts = 0;
const MAX_ATTEMPTS = 3;

// ── Fusion & Decision Module (matches Figure 9) ───────────
// Scores are stored when a modality FAILS its individual threshold.
// If both fail, the Fusion & Decision Module combines them.
// Only if fusion ALSO fails does the system escalate to security question.
//
// Weights: keystroke 0.45, voice 0.55
// Fusion threshold: 0.50 — must collectively clear this to grant access
//
// Flow (Figure 9):
//   Keystroke → high confidence? yes → GRANT
//                               no  → Voice → high confidence? yes → GRANT
//                                                               no  → Fusion → high confidence? yes → GRANT
//                                                                              no  → Security Question

const FUSION_WEIGHT_KS    = 0.45;
const FUSION_WEIGHT_VOICE = 0.55;
const FUSION_THRESHOLD    = 0.50;

let _ksScore    = null;   // stored when keystroke fails individual threshold
let _voiceScore = null;   // stored when voice fails individual threshold


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
    document.getElementById("password-section").classList.remove("hidden");
    document.getElementById("attempts-indicator").classList.remove("hidden");

    document.getElementById("dot-password")
        .classList.replace("bg-gray-700", "bg-purple-600");
}


// ── Step 1: Password Auth ─────────────────────────────────
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
function moveToKeystrokeAuth() {
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
            // Keystroke individually passed — grant immediately (Figure 9: yes branch)
            showSuccess("Keystroke Dynamics", result.confidence);
        } else {
            // Keystroke failed individual threshold — store score for fusion later
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

        // Normalise voice score to 0–1
        const voiceScore = result.fused_score != null
            ? result.fused_score / 100.0
            : (typeof result.confidence === "number" ? result.confidence : 0.0);

        if (result.authenticated) {
            // Voice individually passed — grant immediately (Figure 9: yes branch)
            showSuccess("Speech Biometrics", result.confidence);

        } else {
            // Voice failed individual threshold — store score for fusion
            _voiceScore = voiceScore;
            recordFailedAttempt();

            // ── Fusion & Decision Module (Figure 9) ───────────────
            // Both keystroke AND voice have now failed individually.
            // Run the Fusion & Decision Module before escalating to security question.
            const ksScore = _ksScore !== null ? _ksScore : 0.0;
            const fusedScore = (FUSION_WEIGHT_KS * ksScore) + (FUSION_WEIGHT_VOICE * _voiceScore);

            console.log(
                `[Fusion & Decision Module] ` +
                `keystroke=${(ksScore * 100).toFixed(1)}%  ` +
                `voice=${(_voiceScore * 100).toFixed(1)}%  ` +
                `fused=${(fusedScore * 100).toFixed(1)}%  ` +
                `threshold=${(FUSION_THRESHOLD * 100).toFixed(0)}%`
            );

            if (fusedScore >= FUSION_THRESHOLD) {
                // Fusion passed — grant access (Figure 9: fusion yes branch)
                showSuccess("Fusion & Decision Module", fusedScore);
            } else {
                // Fusion also failed — escalate to security question (Figure 9: fusion no branch)
                const voicePct  = (_voiceScore * 100).toFixed(1);
                const fusedPct  = (fusedScore * 100).toFixed(1);
                status.textContent =
                    `❌ Voice not matched (${voicePct}%) — fused score ${fusedPct}% — proceeding to security question…`;
                status.className = "text-center text-sm mb-4 text-red-400";
                setTimeout(() => moveToSecurityAuth(), 2000);
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
    document.getElementById("line-3").style.width = "100%";

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
            // Correct answer → re-authenticate from the beginning (Figure 9)
            // The security question confirms identity but does not grant access directly.
            // The user must pass keystroke or voice on a fresh attempt.
            status.textContent = "✅ Correct! Please re-authenticate with your biometrics.";
            status.className   = "text-center text-sm mb-4 text-green-400";
            setTimeout(() => {
                // Reset scores and failed attempts, then restart from keystroke
                _ksScore       = null;
                _voiceScore    = null;
                failedAttempts = 0;
                [1, 2, 3].forEach(i => {
                    const dot = document.getElementById(`fail-${i}`);
                    if (dot) dot.classList.replace("bg-red-500", "bg-gray-700");
                });
                moveToKeystrokeAuth();
            }, 1500);
        } else {
            // Wrong answer → flagged and rejected (Figure 9)
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
        "username-section", "password-section",
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

    ["dot-password", "dot-keystroke",
     "dot-voice", "dot-security"].forEach(id => {
        const el = document.getElementById(id);
        if (el) el.classList.replace("bg-purple-600", "bg-gray-700");
    });

    document.getElementById("line-1").style.width = "0%";
    document.getElementById("line-2").style.width = "0%";
    document.getElementById("line-3").style.width = "0%";

    hideAllSections();
    document.getElementById("username-section").classList.remove("hidden");
    document.getElementById("step-indicator").classList.add("hidden");
    document.getElementById("attempts-indicator").classList.add("hidden");
    document.getElementById("username-input").value = "";
}