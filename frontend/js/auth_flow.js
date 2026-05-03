// frontend/js/auth_flow.js

let authUsername   = "";
// Kept in memory for the single request to /auth/security-question, which
// requires password re-verification to avoid leaking the question for any
// email. Cleared on resetLogin().
let authPassword   = "";
let failedAttempts = 0;
const MAX_ATTEMPTS = 3;

// ── Fusion & Decision Module ──────────────────────────────
// Scores are sent to the backend /auth/fuse endpoint for the authoritative
// grant/deny decision. No fusion math is done here — the server decides.
let _ksScore      = null;
let _ksPassed     = false;
let _voiceScore   = null;
let _voicePassed  = false;


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
            authPassword = password;
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

    // Clear any leftover text from previous attempt and re-enable the input
    // (it's disabled on submit, so a re-auth round must explicitly re-enable it).
    const input = document.getElementById("keystroke-input");
    if (input) {
        input.value = "";
        input.disabled = false;
    }

    KeystrokeCapture.reset();
    KeystrokeCapture.attach("keystroke-input");
    document.getElementById("keystroke-status").textContent = "Start typing when ready";

    // TypingDNA-style auto-submit: fire submitKeystrokeAuth() as soon as the
    // typed phrase matches the target. 120ms delay lets the final keyup land
    // in KeystrokeCapture before features are extracted.
    if (input) {
        let _submitPending = false;
        input.onkeyup = () => {
            if (_submitPending) return;
            const trimmed = input.value.trim();
            const target  = KeystrokeCapture.targetPhrase;
            if (trimmed === target) {
                _submitPending = true;
                input.value    = target;
                setTimeout(() => {
                    input.disabled = true;
                    submitKeystrokeAuth();
                }, 120);
            }
        };
    }
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

        _ksScore  = typeof result.confidence === "number" ? result.confidence : 0.0;
        _ksPassed = result.authenticated === true;

        // ── Adaptive fusion decision ──────────────────────────────────────
        // HIGH confidence (≥ 0.90): keystroke alone is sufficient — grant immediately.
        //
        // UNCERTAIN (0.55–0.89): keystroke passed its per-user threshold but
        //   confidence isn't top-tier — voice confirms and backend fuses both.
        //
        // LOW (< 0.55): keystroke clearly failed — voice still runs but the
        //   backend applies a stricter fusion threshold.
        const KS_HIGH_CONF = 0.90;

        if (_ksScore >= KS_HIGH_CONF) {
            // High-confidence single-modality grant — no voice needed
            showSuccess("Keystroke Dynamics", _ksScore);
        } else if (_ksPassed) {
            // Passed threshold but not high-confidence — add voice for fusion
            status.textContent =
                `✅ Keystroke matched — confirming with voice…`;
            status.className = "text-center text-sm mb-4 text-green-400";
            setTimeout(() => moveToVoiceAuth(), 1200);
        } else {
            // Low keystroke confidence — voice needed for fusion decision
            recordFailedAttempt();
            status.textContent =
                `⚠️ Keystroke uncertain — confirming with voice…`;
            status.className = "text-center text-sm mb-4 text-yellow-400";
            setTimeout(() => moveToVoiceAuth(), 1200);
        }

    } catch (err) {
        console.error(err);
        status.textContent = "❌ Server error. Try again.";
        status.className   = "text-center text-sm mb-4 text-red-400";
    }
}


// ── Step 2: Voice Auth ────────────────────────────────────

// Fetch a fresh server-issued challenge (random word order) and update the
// voice-prompt display. The backend pops the issued challenge on every
// /auth/voice POST, so retries MUST refetch — otherwise the server falls back
// to the canonical stored phrase order while the UI still shows the prior
// random order, causing a guaranteed mismatch.
async function refreshVoiceChallenge() {
    const voicePrompts = document.querySelectorAll("#voice-section .phrase-display");
    const priorTexts = Array.from(voicePrompts).map(el => el.textContent);
    voicePrompts.forEach(el => { el.textContent = "Loading prompt…"; });
    try {
        const res = await Api.getVoiceChallenge(authUsername);
        const challenge = res && res.challenge ? res.challenge : "";
        if (challenge) {
            voicePrompts.forEach(el => { el.textContent = challenge; });
        } else {
            voicePrompts.forEach((el, i) => { el.textContent = priorTexts[i] || ""; });
        }
    } catch (e) {
        console.warn("Could not fetch voice challenge:", e);
        voicePrompts.forEach((el, i) => { el.textContent = priorTexts[i] || ""; });
    }
}

async function moveToVoiceAuth() {
    hideAllSections();
    document.getElementById("step-indicator").classList.remove("hidden");
    document.getElementById("voice-section").classList.remove("hidden");
    document.getElementById("attempts-indicator").classList.remove("hidden");

    document.getElementById("dot-voice")
        .classList.replace("bg-gray-700", "bg-purple-600");
    document.getElementById("line-1").style.width = "100%";

    await refreshVoiceChallenge();
}

function startVoiceAuth() {
    // Delegates entirely to startRecording() in speech.js.
    // VAD (Silero) handles start/stop automatically — no manual button wiring needed.
    startRecording();
}

// Resets the voice step so the user can try a fresh recording
function resetVoiceAuth() {
    // Refresh the challenge so the displayed prompt matches what the server
    // will check on the next /auth/voice POST (each POST pops the prior one).
    refreshVoiceChallenge();
    SpeechCapture.reset().then(() => {
        const btn      = document.getElementById("record-btn");
        const tryAgain = document.getElementById("try-again-btn");
        const diagText = document.getElementById("diag-text");
        const status   = document.getElementById("voice-status");

        if (btn) {
            btn.disabled    = false;
            btn.textContent = "🎤 Start Recording";
            btn.onclick     = startVoiceAuth;
        }
        if (tryAgain) tryAgain.disabled = false;
        if (diagText) diagText.textContent = "";
        if (status) {
            status.textContent = "Click record when ready";
            status.className   = "text-center text-sm mb-4 text-gray-500";
        }
    });
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

        // Server's issued challenge wasn't found (TTL expired, server reload,
        // or already popped). Don't count this as a failed attempt — refetch
        // a fresh challenge and have the user try again. Without this branch
        // the server used to silently fall back to the canonical phrase
        // order, so the displayed prompt and the server-checked phrase would
        // diverge and legit users could never succeed on that session.
        if (result.result === "challenge_expired") {
            status.textContent = "⚠ Voice prompt expired — please record again.";
            status.className   = "text-center text-sm mb-4 text-yellow-400";
            if (btn) { btn.disabled = false; btn.textContent = "🎤 Try Again"; btn.onclick = resetVoiceAuth; }
            if (tryAgain) tryAgain.disabled = false;
            refreshVoiceChallenge();
            return;
        }

        _voiceScore  = typeof result.confidence === "number" ? result.confidence : 0.0;
        _voicePassed = result.authenticated === true;

        // Show specific phrase error if voice matched but wrong phrase was spoken
        if (!_voicePassed && result.phrase_error) {
            status.textContent = `❌ ${result.phrase_error}`;
            status.className   = "text-center text-sm mb-4 text-red-400";
            if (btn) { btn.disabled = false; btn.textContent = "🎤 Try Again"; btn.onclick = resetVoiceAuth; }
            if (tryAgain) tryAgain.disabled = false;
            // Server already popped the previous challenge — refresh so the
            // displayed prompt matches what the next /auth/voice will check.
            refreshVoiceChallenge();
            return;
        }

        status.textContent = "⏳ Running fusion decision…";
        status.className   = "text-center text-sm mb-4 text-yellow-400";

        // ── Backend Fusion & Decision ─────────────────────────
        // The server makes the authoritative grant/deny decision.
        // No grant/deny logic runs in the browser.
        const ksScore = _ksScore !== null ? _ksScore : 0.0;
        const fuseResult = await Api.fuseScores(
            authUsername, ksScore, _voiceScore, _ksPassed, _voicePassed
        );

        console.log(
            `[Fusion] keystroke=${(ksScore * 100).toFixed(1)}%  ` +
            `voice=${(_voiceScore * 100).toFixed(1)}%  ` +
            `fused=${(fuseResult.fused_score * 100).toFixed(1)}%  ` +
            `→ ${fuseResult.granted ? "GRANT" : "DENY"}`
        );

        if (fuseResult.granted) {
            showSuccess("Fusion & Decision Module", fuseResult.fused_score);
        } else {
            recordFailedAttempt();
            status.textContent =
                `❌ Additional verification needed — proceeding to security question…`;
            status.className = "text-center text-sm mb-4 text-red-400";
            setTimeout(() => moveToSecurityAuth(), 1000);
        }

    } catch (err) {
        console.error(err);
        status.textContent = "❌ Server error. Try again.";
        status.className   = "text-center text-sm mb-4 text-red-400";
        if (btn) { btn.disabled = false; btn.textContent = "🎤 Try Again"; btn.onclick = resetVoiceAuth; }
        if (tryAgain) tryAgain.disabled = false;
        refreshVoiceChallenge();
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

    const _sqInput = document.getElementById("security-answer-input");
    _sqInput.value = "";
    setTimeout(() => { _sqInput.value = ""; }, 50);

    document.getElementById("dot-security")
        .classList.replace("bg-gray-700", "bg-purple-600");
    document.getElementById("line-2").style.width = "100%";

    try {
        const result = await Api.getSecurityQuestion(authUsername, authPassword);
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
        const result = await Api.verifySecurityQuestion(authUsername, answer, _ksScore);

        if (result.authenticated) {
            // Injury exemption: voice already passed this session but
            // keystroke failed → typing is affected, voice isn't, so grant
            // directly. Any other case → re-authenticate with keystroke + voice.
            const injuryExempt = _voicePassed && !_ksPassed;

            if (injuryExempt) {
                status.textContent =
                    `✅ Identity confirmed (injury fallback — voice biometric verified).`;
                status.className = "text-center text-sm mb-4 text-green-400";
                setTimeout(() => {
                    showSuccess("Security Question + Voice Biometric", _voiceScore);
                }, 800);
                return;
            }

            status.textContent =
                "✅ Identity confirmed. Please re-authenticate with keystroke and voice.";
            status.className = "text-center text-sm mb-4 text-yellow-400";

            _ksScore     = null;
            _ksPassed    = false;
            _voiceScore  = null;
            _voicePassed = false;

            setTimeout(() => {
                document.getElementById("security-answer-input").value = "";
                moveToKeystrokeAuth();
            }, 1500);
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
    // Confidence display intentionally omitted — not shown to end users.
    const confEl = document.getElementById("success-confidence");
    if (confEl) confEl.textContent = "";
    // Persist session context for the dashboard's re-auth flow.
    try {
        sessionStorage.setItem("authUsername", authUsername || "");
        sessionStorage.setItem("authLoginAt", Date.now().toString());
        sessionStorage.setItem("authLastMethod", method);
        sessionStorage.setItem("authLastConfidence", String(confidence));
    } catch (_) {}
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
    authPassword   = "";
    _ksScore       = null;
    _ksPassed      = false;
    _voiceScore    = null;
    _voicePassed   = false;
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