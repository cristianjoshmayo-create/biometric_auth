// frontend/js/enroll.js

let currentUsername = "";
let currentPassword = "";  // ← ADDED

let currentKeystrokeAttempt = 1;
const KEYSTROKE_TARGET = 3;

let voiceAttemptsSaved = 0;
const VOICE_TARGET = 3;

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
function moveToKeystrokeEnrollment() {  // ← ADDED (was previously inline in startEnrollment)
    document.getElementById("password-section").classList.add("hidden");
    document.getElementById("keystroke-section").classList.remove("hidden");

    document.getElementById("step2-dot").querySelector("div")
        .classList.replace("bg-gray-700", "bg-purple-600");
    document.getElementById("progress-line").style.width = "100%";

    KeystrokeCapture.attach("keystroke-input");
    document.getElementById("keystroke-status").textContent = "Start typing when ready";
}

async function submitKeystroke() {
    const input  = document.getElementById("keystroke-input");
    const status = document.getElementById("keystroke-status");

    if (!KeystrokeCapture.validatePhrase(input.value)) {
        status.textContent = "❌ Phrase doesn't match. Please type exactly as shown.";
        status.className = "text-center text-sm mb-4 text-red-400";
        input.value = "";
        KeystrokeCapture.reset();
        KeystrokeCapture.attach("keystroke-input");
        return;
    }

    const features = KeystrokeCapture.extractFeatures();
    if (!features) {
        status.textContent = "❌ Not enough keystroke data. Try again.";
        status.className = "text-center text-sm mb-4 text-red-400";
        return;
    }

    status.textContent = `⏳ Saving attempt ${currentKeystrokeAttempt}/${KEYSTROKE_TARGET}…`;
    status.className = "text-center text-sm mb-4 text-yellow-400";

    try {
        const result = await Api.enrollKeystroke(currentUsername, {
            dwell_times:  features.dwell_times,
            flight_times: features.flight_times,
            typing_speed: features.typing_speed,
            dwell_mean:    features.dwell_mean,
            dwell_std:     features.dwell_std,
            dwell_median:  features.dwell_median,
            dwell_min:     features.dwell_min,
            dwell_max:     features.dwell_max,
            flight_mean:   features.flight_mean,
            flight_std:    features.flight_std,
            flight_median: features.flight_median,
            p2p_mean:      features.p2p_mean,
            p2p_std:       features.p2p_std,
            r2r_mean:      features.r2r_mean,
            r2r_std:       features.r2r_std,
            digraph_th: features.digraph_th || 0,
            digraph_he: features.digraph_he || 0,
            digraph_bi: features.digraph_bi || 0,
            digraph_io: features.digraph_io || 0,
            digraph_om: features.digraph_om || 0,
            digraph_me: features.digraph_me || 0,
            digraph_et: features.digraph_et || 0,
            digraph_tr: features.digraph_tr || 0,
            digraph_ri: features.digraph_ri || 0,
            digraph_ic: features.digraph_ic || 0,
            digraph_vo: features.digraph_vo || 0,
            digraph_oi: features.digraph_oi || 0,
            digraph_ce: features.digraph_ce || 0,
            digraph_ke: features.digraph_ke || 0,
            digraph_ey: features.digraph_ey || 0,
            digraph_ys: features.digraph_ys || 0,
            digraph_st: features.digraph_st || 0,
            digraph_ro: features.digraph_ro || 0,
            digraph_ok: features.digraph_ok || 0,
            digraph_au: features.digraph_au || 0,
            digraph_ut: features.digraph_ut || 0,
            digraph_en: features.digraph_en || 0,
            digraph_nt: features.digraph_nt || 0,
            digraph_ti: features.digraph_ti || 0,
            digraph_ca: features.digraph_ca || 0,
            digraph_at: features.digraph_at || 0,
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
            status.className = "text-center text-sm mb-4 text-red-400";
            return;
        }

        const dot = document.getElementById(`attempt-${currentKeystrokeAttempt}`);
        if (dot) dot.classList.replace("bg-gray-700", "bg-purple-600");

        status.textContent = `✅ Attempt ${currentKeystrokeAttempt}/${KEYSTROKE_TARGET} saved!`;
        status.className = "text-center text-sm mb-4 text-green-400";

        currentKeystrokeAttempt++;

        if (currentKeystrokeAttempt <= KEYSTROKE_TARGET) {
            const label = document.getElementById("attempt-label");
            if (label) label.textContent = `Attempt ${currentKeystrokeAttempt} of ${KEYSTROKE_TARGET}`;

            input.value = "";
            KeystrokeCapture.reset();
            KeystrokeCapture.attach("keystroke-input");

            setTimeout(() => {
                status.textContent = "Start typing when ready";
                status.className = "text-center text-sm mb-4 text-gray-400";
            }, 1000);
        } else {
            status.textContent = `✅ All ${KEYSTROKE_TARGET} attempts saved! Moving on…`;
            setTimeout(() => moveToVoiceEnrollment(), 1000);
        }

    } catch (err) {
        status.textContent = "❌ Network error — check server is running.";
        status.className = "text-center text-sm mb-4 text-red-400";
        console.error("Keystroke enroll error:", err);
    }
}

// ── Step 3: Voice Enrollment ──────────────────────────────────────────────
function moveToVoiceEnrollment() {
    document.getElementById("keystroke-section").classList.add("hidden");
    document.getElementById("voice-section").classList.remove("hidden");

    document.getElementById("step3-dot").querySelector("div")
        .classList.replace("bg-gray-700", "bg-purple-600");
    document.getElementById("progress-line2").style.width = "100%";

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