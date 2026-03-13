// frontend/js/enroll.js

let currentUsername = "";
let currentPassword = "";

let currentKeystrokeAttempt = 1;
const KEYSTROKE_TARGET = 5;

// Quality tracking across attempts
const attemptSpeeds = [];

let voiceAttemptsSaved = 0;
const VOICE_TARGET = 3;

// ── Enrollment state persistence ──────────────────────────────────────────
// Saves progress to sessionStorage so a server restart (uvicorn --reload
// triggered by .pkl file writes) doesn't snap the user back to step 1.
const ENROLL_STATE_KEY = "biometric_enroll_state";

function saveEnrollState(section) {
    sessionStorage.setItem(ENROLL_STATE_KEY, JSON.stringify({
        section,
        username:               currentUsername,
        keystrokeAttempt:       currentKeystrokeAttempt,
        voiceAttemptsSaved:     voiceAttemptsSaved,
    }));
}

function clearEnrollState() {
    sessionStorage.removeItem(ENROLL_STATE_KEY);
}

function restoreEnrollState() {
    const raw = sessionStorage.getItem(ENROLL_STATE_KEY);
    if (!raw) return false;
    try {
        const s = JSON.parse(raw);
        if (!s.username) return false;

        currentUsername           = s.username;
        currentKeystrokeAttempt   = s.keystrokeAttempt   || 1;
        voiceAttemptsSaved        = s.voiceAttemptsSaved  || 0;

        // Hide username section, show step indicator
        document.getElementById("username-section").classList.add("hidden");
        document.getElementById("step-indicator").classList.remove("hidden");
        document.getElementById("attempts-indicator")?.classList.remove("hidden");

        // Restore step dots up to current section
        const stepOrder = ["password", "keystroke", "voice", "security", "success"];
        const idx = stepOrder.indexOf(s.section);
        for (let i = 0; i <= idx && i < 4; i++) {
            const dot = document.getElementById(`step${i+1}-dot`);
            if (dot) dot.querySelector("div").classList.replace("bg-gray-700", "bg-purple-600");
        }
        if (idx >= 1) document.getElementById("progress-line").style.width  = "100%";
        if (idx >= 2) document.getElementById("progress-line2").style.width = "100%";
        if (idx >= 3) document.getElementById("progress-line3").style.width = "100%";

        // Jump straight to the right section
        if (s.section === "keystroke") {
            document.getElementById("keystroke-section").classList.remove("hidden");
            _resetKeystrokeInput();
            _updateKeystrokeProgress();
        } else if (s.section === "voice") {
            document.getElementById("voice-section").classList.remove("hidden");
            updateVoiceAttemptUI(voiceAttemptsSaved);
        } else if (s.section === "security") {
            document.getElementById("security-section").classList.remove("hidden");
        } else if (s.section === "success") {
            document.getElementById("success-section").classList.remove("hidden");
        } else {
            // password section — just show it
            document.getElementById("password-section").classList.remove("hidden");
        }

        console.log(`[enroll] Restored state: section=${s.section} user=${s.username}`);
        return true;
    } catch(e) {
        sessionStorage.removeItem(ENROLL_STATE_KEY);
        return false;
    }
}

// Attempt restore on page load
document.addEventListener("DOMContentLoaded", () => {
    restoreEnrollState();
});

// ── Step 0: Start enrollment ──────────────────────────────────────────────
function startEnrollment() {
    const username = document.getElementById("username-input").value.trim();
    if (!username) {
        alert("Please enter a username first.");
        return;
    }

    currentUsername = username;

    document.getElementById("username-section").classList.add("hidden");
    document.getElementById("step-indicator").classList.remove("hidden");
    document.getElementById("password-section").classList.remove("hidden");  // ← UPDATED

    document.getElementById("step1-dot").querySelector("div")
        .classList.replace("bg-gray-700", "bg-purple-600");
}

// ── Step 1: Password Enrollment ← ADDED ──────────────────────────────────
async function submitPassword() {
    const password = document.getElementById("password-input").value;
    const confirm  = document.getElementById("password-confirm").value;
    const status   = document.getElementById("password-status");

    if (password.length < 8) {
        status.textContent = "❌ Password must be at least 8 characters.";
        status.className   = "text-center text-sm mb-4 text-red-400";
        return;
    }
    if (password !== confirm) {
        status.textContent = "❌ Passwords do not match.";
        status.className   = "text-center text-sm mb-4 text-red-400";
        return;
    }

    status.textContent = "⏳ Saving...";
    status.className   = "text-center text-sm mb-4 text-yellow-400";

    try {
        const result = await Api.enrollUser(currentUsername, password);  // ← UPDATED

        if (result.success) {
            currentPassword = password;
            status.textContent = "✅ Account created!";
            status.className   = "text-center text-sm mb-4 text-green-400";
            setTimeout(() => moveToKeystrokeEnrollment(), 800);
        } else {
            status.textContent = "❌ " + (result.detail || "Failed.");
            status.className   = "text-center text-sm mb-4 text-red-400";
        }
    } catch (err) {
        status.textContent = "❌ Network error.";
        status.className   = "text-center text-sm mb-4 text-red-400";
        console.error("Password enroll error:", err);
    }
}

// ── Step 2: Keystroke Enrollment ──────────────────────────────────────────
function moveToKeystrokeEnrollment() {
    document.getElementById("password-section").classList.add("hidden");
    document.getElementById("keystroke-section").classList.remove("hidden");

    document.getElementById("step2-dot").querySelector("div")
        .classList.replace("bg-gray-700", "bg-purple-600");
    document.getElementById("progress-line").style.width = "100%";

    saveEnrollState("keystroke");
    _resetKeystrokeInput();
}

function _resetKeystrokeInput() {
    const input = document.getElementById("keystroke-input");
    input.value = "";
    input.disabled = false;
    input.focus();

    KeystrokeCapture.reset();
    KeystrokeCapture.attach("keystroke-input");

    _updateKeystrokeProgress();

    // Auto-submit when phrase is completed — no button needed
    input.oninput = () => {
        const val = input.value;

        // Live character-match colour feedback
        const target = KeystrokeCapture.targetPhrase;
        if (val.length > 0 && !target.startsWith(val)) {
            input.classList.add("border-red-500");
            input.classList.remove("border-gray-700", "border-green-500");
        } else if (val.length === target.length && val === target) {
            input.classList.add("border-green-500");
            input.classList.remove("border-gray-700", "border-red-500");
        } else {
            input.classList.remove("border-red-500", "border-green-500");
            input.classList.add("border-gray-700");
        }

        // Auto-submit once the full phrase is typed correctly
        if (val === target) {
            input.disabled = true;
            setTimeout(() => submitKeystroke(), 150); // tiny delay so last keyup fires
        }
    };
}

function _updateKeystrokeProgress() {
    const done  = currentKeystrokeAttempt - 1;
    const pct   = Math.round((done / KEYSTROKE_TARGET) * 100);
    const label = document.getElementById("attempt-label");
    const bar   = document.getElementById("keystroke-progress-bar");
    const pctEl = document.getElementById("attempt-pct");

    if (bar)   bar.style.width = `${pct}%`;
    if (pctEl) pctEl.textContent = `${pct}%`;

    if (label) {
        const remaining = KEYSTROKE_TARGET - done;
        if (done === 0) {
            label.textContent = `Attempt 1 of ${KEYSTROKE_TARGET}`;
        } else if (remaining === 1) {
            label.textContent = `Almost there! Last attempt`;
        } else if (remaining > 0) {
            label.textContent = `${remaining} more to go…`;
        } else {
            label.textContent = `All ${KEYSTROKE_TARGET} attempts done`;
        }
    }
}

async function submitKeystroke() {
    const input  = document.getElementById("keystroke-input");
    const status = document.getElementById("keystroke-status");

    if (!KeystrokeCapture.validatePhrase(input.value)) {
        status.textContent = "❌ Phrase doesn't match — please type exactly as shown.";
        status.className = "text-center text-sm mb-2 text-red-400";
        input.disabled = false;
        input.value = "";
        input.focus();
        KeystrokeCapture.reset();
        KeystrokeCapture.attach("keystroke-input");
        input.oninput = arguments.callee; // re-attach auto-submit
        _resetKeystrokeInput();
        return;
    }

    const features = KeystrokeCapture.extractFeatures();
    if (!features) {
        status.textContent = "❌ Not enough keystroke data — try again.";
        status.className = "text-center text-sm mb-2 text-red-400";
        _resetKeystrokeInput();
        return;
    }

    status.textContent = `⏳ Saving…`;
    status.className = "text-center text-sm mb-2 text-yellow-400";

    try {
        const result = await Api.enrollKeystroke(currentUsername, {
            dwell_times:  features.dwell_times,
            flight_times: features.flight_times,
            typing_speed: features.typing_speed,
            dwell_mean:    features.dwell_mean,    dwell_std:     features.dwell_std,
            dwell_median:  features.dwell_median,  dwell_min:     features.dwell_min,
            dwell_max:     features.dwell_max,
            flight_mean:   features.flight_mean,   flight_std:    features.flight_std,
            flight_median: features.flight_median,
            p2p_mean:      features.p2p_mean,      p2p_std:       features.p2p_std,
            r2r_mean:      features.r2r_mean,       r2r_std:       features.r2r_std,
            digraph_th: features.digraph_th || 0, digraph_he: features.digraph_he || 0,
            digraph_bi: features.digraph_bi || 0, digraph_io: features.digraph_io || 0,
            digraph_om: features.digraph_om || 0, digraph_me: features.digraph_me || 0,
            digraph_et: features.digraph_et || 0, digraph_tr: features.digraph_tr || 0,
            digraph_ri: features.digraph_ri || 0, digraph_ic: features.digraph_ic || 0,
            digraph_vo: features.digraph_vo || 0, digraph_oi: features.digraph_oi || 0,
            digraph_ce: features.digraph_ce || 0, digraph_ke: features.digraph_ke || 0,
            digraph_ey: features.digraph_ey || 0, digraph_ys: features.digraph_ys || 0,
            digraph_st: features.digraph_st || 0, digraph_ro: features.digraph_ro || 0,
            digraph_ok: features.digraph_ok || 0, digraph_au: features.digraph_au || 0,
            digraph_ut: features.digraph_ut || 0, digraph_en: features.digraph_en || 0,
            digraph_nt: features.digraph_nt || 0, digraph_ti: features.digraph_ti || 0,
            digraph_ca: features.digraph_ca || 0, digraph_at: features.digraph_at || 0,
            digraph_on: features.digraph_on || 0,
            typing_speed_cpm:        features.typing_speed_cpm,
            typing_duration:         features.typing_duration,
            rhythm_mean:             features.rhythm_mean,
            rhythm_std:              features.rhythm_std,
            rhythm_cv:               features.rhythm_cv,
            pause_count:             features.pause_count,
            pause_mean:              features.pause_mean,
            backspace_ratio:         features.backspace_ratio,
            backspace_count:         features.backspace_count,
            hand_alternation_ratio:  features.hand_alternation_ratio,
            same_hand_sequence_mean: features.same_hand_sequence_mean,
            finger_transition_ratio: features.finger_transition_ratio,
            seek_time_mean:          features.seek_time_mean,
            seek_time_count:         features.seek_time_count,
            shift_lag_mean:   features.shift_lag_mean   || 0,
            shift_lag_std:    features.shift_lag_std    || 0,
            shift_lag_count:  features.shift_lag_count  || 0,
            dwell_mean_norm:  features.dwell_mean_norm  || 0,
            dwell_std_norm:   features.dwell_std_norm   || 0,
            flight_mean_norm: features.flight_mean_norm || 0,
            flight_std_norm:  features.flight_std_norm  || 0,
            p2p_std_norm:     features.p2p_std_norm     || 0,
            r2r_mean_norm:    features.r2r_mean_norm    || 0,
            shift_lag_norm:   features.shift_lag_norm   || 0,
        });

        if (!result.success) {
            status.textContent = "❌ Save failed: " + (result.detail || "unknown error");
            status.className = "text-center text-sm mb-2 text-red-400";
            _resetKeystrokeInput();
            return;
        }

        // ── Quality feedback ─────────────────────────────────────────────
        attemptSpeeds.push(features.typing_speed_cpm);
        let qualityNote = "";
        if (features.typing_speed_cpm < 80) {
            qualityNote = "Try to type at your natural pace next time.";
        } else if (features.typing_speed_cpm > 500) {
            qualityNote = "Great speed! Keep it natural.";
        } else if (attemptSpeeds.length >= 2) {
            const avg = attemptSpeeds.reduce((a, b) => a + b) / attemptSpeeds.length;
            const sdv = Math.sqrt(attemptSpeeds.reduce((s, v) => s + (v - avg) ** 2, 0) / attemptSpeeds.length);
            qualityNote = sdv < 40 ? "Consistent — great!" : "Try to keep a steady pace.";
        }

        // Update live feedback row
        const speedEl = document.getElementById("ks-speed-display");
        const consEl  = document.getElementById("ks-consistency-display");
        if (speedEl) speedEl.textContent = `Speed: ${Math.round(features.typing_speed_cpm)} cpm`;
        if (consEl && attemptSpeeds.length >= 2) {
            const avg = attemptSpeeds.reduce((a, b) => a + b) / attemptSpeeds.length;
            const sdv = Math.sqrt(attemptSpeeds.reduce((s, v) => s + (v - avg) ** 2, 0) / attemptSpeeds.length);
            consEl.textContent = `Consistency: ${sdv < 40 ? "✓ Good" : "± Varies"}`;
        }

        currentKeystrokeAttempt++;
        saveEnrollState("keystroke");
        _updateKeystrokeProgress();

        if (currentKeystrokeAttempt <= KEYSTROKE_TARGET) {
            status.textContent = `✅ Saved! ${qualityNote}`;
            status.className = "text-center text-sm mb-2 text-green-400";

            // Auto-reset for next attempt after a short pause
            setTimeout(() => {
                status.textContent = "Type the phrase again at your natural pace";
                status.className = "text-center text-sm mb-2 text-gray-400";
                _resetKeystrokeInput();
            }, 900);
        } else {
            status.textContent = `✅ All ${KEYSTROKE_TARGET} attempts captured — model training…`;
            status.className = "text-center text-sm mb-2 text-green-400";
            setTimeout(() => moveToVoiceEnrollment(), 1200);
        }

    } catch (err) {
        status.textContent = "❌ Network error — check server is running.";
        status.className = "text-center text-sm mb-2 text-red-400";
        console.error("Keystroke enroll error:", err);
        _resetKeystrokeInput();
    }
}

// ── Step 3: Voice Enrollment ──────────────────────────────────────────────
function moveToVoiceEnrollment() {
    document.getElementById("keystroke-section").classList.add("hidden");
    document.getElementById("voice-section").classList.remove("hidden");

    document.getElementById("step3-dot").querySelector("div")
        .classList.replace("bg-gray-700", "bg-purple-600");
    document.getElementById("progress-line2").style.width = "100%";

    saveEnrollState("voice");
    updateVoiceAttemptUI(0);
}

async function onVoiceRecorded(fullFeatureDict) {
    const status = document.getElementById("voice-status");

    if (voiceAttemptsSaved >= VOICE_TARGET) return;

    status.textContent = `⏳ Saving recording ${voiceAttemptsSaved + 1}/${VOICE_TARGET}…`;
    status.className   = "text-center text-sm mb-4 text-yellow-400";

    try {
        const result = await Api.enrollVoice(currentUsername, fullFeatureDict);

        if (!result.success) {
            status.textContent = "❌ Save failed: " + (result.detail || "unknown error");
            status.className   = "text-center text-sm mb-4 text-red-400";
            return;
        }

        voiceAttemptsSaved = result.attempt_number;
        saveEnrollState("voice");
        updateVoiceAttemptUI(voiceAttemptsSaved);

        if (voiceAttemptsSaved >= VOICE_TARGET) {
            status.textContent = `✅ All ${VOICE_TARGET} recordings saved!`;
            status.className   = "text-center text-sm mb-4 text-green-400";

            const btn = document.getElementById("record-btn");
            if (btn) {
                btn.disabled    = true;
                btn.textContent = `✅ ${VOICE_TARGET}/${VOICE_TARGET} Complete`;
            }

            setTimeout(() => moveToSecurityQuestion(), 1200);
        } else {
            status.textContent =
                `✅ Recording ${voiceAttemptsSaved}/${VOICE_TARGET} saved. Record next.`;
            status.className   = "text-center text-sm mb-4 text-green-400";

            const btn = document.getElementById("record-btn");
            if (btn) {
                btn.disabled    = false;
                btn.textContent = `🎤 Record ${voiceAttemptsSaved + 1}/${VOICE_TARGET}`;
            }
        }

    } catch (err) {
        status.textContent = "❌ Network error — check server is running.";
        status.className   = "text-center text-sm mb-4 text-red-400";
        console.error("Voice enroll save error:", err);
    }
}

function updateVoiceAttemptUI(count) {
    for (let i = 1; i <= VOICE_TARGET; i++) {
        const dot = document.getElementById(`vattempt-${i}`);
        if (dot) {
            if (i <= count) {
                dot.classList.replace("bg-gray-700", "bg-purple-600");
            } else {
                dot.classList.replace("bg-purple-600", "bg-gray-700");
            }
        }
    }

    const label = document.getElementById("vattempt-label");
    if (label) {
        label.textContent = count < VOICE_TARGET
            ? `Recording ${count + 1} of ${VOICE_TARGET}`
            : `All ${VOICE_TARGET} recordings complete`;
    }
}

// ── Step 4: Security Question ─────────────────────────────────────────────
function moveToSecurityQuestion() {
    document.getElementById("voice-section").classList.add("hidden");
    document.getElementById("security-section").classList.remove("hidden");

    document.getElementById("step4-dot").querySelector("div")
        .classList.replace("bg-gray-700", "bg-purple-600");
    document.getElementById("progress-line3").style.width = "100%";

    saveEnrollState("security");
}

async function submitSecurityQuestion() {
    const question = document.getElementById("security-question-select").value;
    const answer   = document.getElementById("security-answer").value.trim();

    if (!question || !answer) {
        alert("Please select a question and provide an answer.");
        return;
    }

    try {
        const result = await Api.enrollSecurity(currentUsername, question, answer);
        if (result.success) {
            clearEnrollState();
            document.getElementById("security-section").classList.add("hidden");
            document.getElementById("success-section").classList.remove("hidden");
        } else {
            alert("Error saving security question: " + (result.detail || "Unknown error"));
        }
    } catch (err) {
        alert("Could not connect to server.");
        console.error(err);
    }
}

function averageArrays(arrays) {
    if (!arrays || arrays.length === 0) return [];
    const minLen = Math.min(...arrays.map(a => a.length));
    const result = [];
    for (let i = 0; i < minLen; i++) {
        const avg = arrays.reduce((sum, arr) => sum + (arr[i] || 0), 0) / arrays.length;
        result.push(parseFloat(avg.toFixed(3)));
    }
    return result;
}