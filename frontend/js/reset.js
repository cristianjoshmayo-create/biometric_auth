// frontend/js/reset.js
// Password-reset page logic. Runs a keystroke + voice biometric challenge,
// then lets the user set a new password.

let resetToken = null;
let resetEmail = "";
let resetPhrase = "";

// ── Section helpers ────────────────────────────────────────────────
function _show(id) {
    ["loading-section", "invalid-section", "keystroke-section",
     "voice-section", "password-section", "success-section", "denied-section"]
        .forEach(s => document.getElementById(s).classList.add("hidden"));
    const el = document.getElementById(id);
    if (el) el.classList.remove("hidden");
}

function _setPhrase(phrase) {
    document.querySelectorAll(".phrase-display").forEach(el => {
        el.textContent = phrase;
    });
    if (typeof KeystrokeCapture !== "undefined" && KeystrokeCapture.setPhrase) {
        KeystrokeCapture.setPhrase(phrase);
    }
}

function _markStep(n) {
    // n: 1 = keystroke done, 2 = voice done, 3 = finished
    if (n >= 1) {
        document.getElementById("line-1").style.width = "100%";
        document.getElementById("dot-2").classList.replace("bg-gray-700", "bg-purple-600");
    }
    if (n >= 2) {
        document.getElementById("line-2").style.width = "100%";
        document.getElementById("dot-3").classList.replace("bg-gray-700", "bg-purple-600");
    }
}

// ── Bootstrap ──────────────────────────────────────────────────────
(async function init() {
    const params = new URLSearchParams(window.location.search);
    resetToken = (params.get("token") || "").trim();
    if (!resetToken) {
        _show("invalid-section");
        return;
    }

    try {
        const info = await Api.resetInfo(resetToken);
        if (!info || !info.email || !info.phrase) {
            _show("invalid-section");
            return;
        }
        resetEmail  = info.email;
        resetPhrase = info.phrase;
        _setPhrase(resetPhrase);

        document.getElementById("header-sub").textContent =
            `Verify your biometrics to reset the password for ${resetEmail}.`;
        document.getElementById("step-indicator").classList.remove("hidden");

        if (info.ks_verified && info.voice_verified) {
            _markStep(2);
            _show("password-section");
        } else if (info.ks_verified) {
            _markStep(1);
            _show("voice-section");
        } else {
            _show("keystroke-section");
            KeystrokeCapture.attach("keystroke-input");
            _wireKeystrokeAutoSubmit();
        }
    } catch (err) {
        console.error("reset info error:", err);
        _show("invalid-section");
    }
})();

// ── Step 1: Keystroke ──────────────────────────────────────────────
async function submitResetKeystroke() {
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
        const result = await Api.resetKeystroke(resetToken, features);
        if (result.reset_ks_verified) {
            status.textContent = "✅ Keystroke verified. Continue to voice.";
            status.className = "text-center text-sm mb-4 text-green-400";
            _markStep(1);
            setTimeout(() => _show("voice-section"), 800);
        } else {
            status.textContent = `❌ Keystroke did not match your enrolled profile.`;
            status.className = "text-center text-sm mb-4 text-red-400";
            // One retry path — user can try typing again, or give up (go to denied)
            setTimeout(() => {
                input.value = "";
                input.disabled = false;
                KeystrokeCapture.reset();
                KeystrokeCapture.attach("keystroke-input");
                _wireKeystrokeAutoSubmit();
                status.textContent = "Try typing once more at your natural pace.";
                status.className = "text-center text-sm mb-4 text-gray-400";
            }, 1800);
        }
    } catch (err) {
        console.error(err);
        status.textContent = "❌ Server error. Try again.";
        status.className = "text-center text-sm mb-4 text-red-400";
    }
}

// TypingDNA-style auto-submit: fire submitResetKeystroke() once typed
// phrase matches target. 120ms delay lets the final keyup land in
// KeystrokeCapture before features are extracted.
function _wireKeystrokeAutoSubmit() {
    const input = document.getElementById("keystroke-input");
    if (!input) return;
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
                submitResetKeystroke();
            }, 120);
        }
    };
}

// ── Step 2: Voice ──────────────────────────────────────────────────
function startResetVoice() {
    if (typeof startRecording === "function") {
        startRecording();
    }
}

// Called by speech.js when it has extracted features
async function onVoiceAuthComplete(fullFeatureDict) {
    const status = document.getElementById("voice-status");
    const btn    = document.getElementById("record-btn");
    status.textContent = "⏳ Verifying voice pattern...";
    status.className = "text-center text-sm mb-4 text-yellow-400";
    if (btn) btn.disabled = true;

    try {
        const result = await Api.resetVoice(resetToken, fullFeatureDict);
        if (result.reset_voice_verified) {
            status.textContent = "✅ Voice verified. Set your new password.";
            status.className = "text-center text-sm mb-4 text-green-400";
            _markStep(2);
            setTimeout(() => _show("password-section"), 900);
        } else {
            const msg = result.phrase_error
                ? `❌ ${result.phrase_error}`
                : `❌ Voice did not match your enrolled profile.`;
            status.textContent = msg;
            status.className = "text-center text-sm mb-4 text-red-400";
            if (btn) {
                btn.disabled = false;
                btn.textContent = "🎤 Try Again";
                btn.onclick = startResetVoice;
            }
        }
    } catch (err) {
        console.error(err);
        status.textContent = "❌ Server error. Try again.";
        status.className = "text-center text-sm mb-4 text-red-400";
        if (btn) { btn.disabled = false; btn.textContent = "🎤 Try Again"; }
    }
}

// ── Step 3: Submit new password ────────────────────────────────────
async function submitResetPassword() {
    const p1 = document.getElementById("new-password").value;
    const p2 = document.getElementById("new-password-confirm").value;
    const status = document.getElementById("password-status");

    if (p1.length < 8) {
        status.textContent = "❌ Password must be at least 8 characters.";
        status.className = "text-center text-sm mb-4 text-red-400";
        return;
    }
    if (p1 !== p2) {
        status.textContent = "❌ Passwords do not match.";
        status.className = "text-center text-sm mb-4 text-red-400";
        return;
    }

    status.textContent = "⏳ Updating password...";
    status.className = "text-center text-sm mb-4 text-yellow-400";

    try {
        const result = await Api.resetSubmit(resetToken, p1);
        if (result.success) {
            _show("success-section");
        } else {
            status.textContent = "❌ " + (result.detail || "Reset failed.");
            status.className = "text-center text-sm mb-4 text-red-400";
        }
    } catch (err) {
        console.error(err);
        status.textContent = "❌ Server error. Try again.";
        status.className = "text-center text-sm mb-4 text-red-400";
    }
}
