// frontend/js/enroll.js

let currentUsername = "";
let currentPassword = "";
let currentPhrase   = "";   // unique 4-word phrase assigned by server at enrollment

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
        phrase:                 currentPhrase,
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
        currentPhrase             = s.phrase || "";
        currentKeystrokeAttempt   = s.keystrokeAttempt   || 1;
        voiceAttemptsSaved        = s.voiceAttemptsSaved  || 0;

        // Restore phrase in UI and KeystrokeCapture
        if (currentPhrase) {
            KeystrokeCapture.setPhrase(currentPhrase);
            document.querySelectorAll(".phrase-display").forEach(el => {
                el.textContent = currentPhrase;
            });
        }

        document.getElementById("username-section").classList.add("hidden");
        document.getElementById("step-indicator").classList.remove("hidden");

        // 3-step system: keystroke=step1, voice=step2, security=step3
        const stepOrder = ["keystroke", "voice", "security", "success"];
        const idx = stepOrder.indexOf(s.section);
        if (idx >= 0) {
            document.getElementById("step1-dot").querySelector("div")
                .classList.replace("bg-gray-700", "bg-purple-600");
        }
        if (idx >= 1) {
            document.getElementById("step2-dot").querySelector("div")
                .classList.replace("bg-gray-700", "bg-purple-600");
            document.getElementById("progress-line").style.width = "100%";
        }
        if (idx >= 2) {
            document.getElementById("step3-dot").querySelector("div")
                .classList.replace("bg-gray-700", "bg-purple-600");
            document.getElementById("progress-line2").style.width = "100%";
        }

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

    document.getElementById("step1-dot").querySelector("div")
        .classList.replace("bg-gray-700", "bg-purple-600");
}

// ── Step 0: Create account (username + password on one screen) ───────────
async function submitPassword() {
    const rawEmail  = document.getElementById("username-input").value.trim();
    const password  = document.getElementById("password-input").value;
    const confirm   = document.getElementById("password-confirm").value;
    const status    = document.getElementById("password-status");

    // Validate email format
    if (!rawEmail) {
        status.textContent = "❌ Please enter your email address.";
        status.className   = "text-center text-sm mb-4 text-red-400";
        return;
    }
    const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
    if (!emailRegex.test(rawEmail)) {
        status.textContent = "❌ Please enter a valid email address (e.g. you@example.com).";
        status.className   = "text-center text-sm mb-4 text-red-400";
        return;
    }
    const email = rawEmail.toLowerCase();

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

    status.textContent = "⏳ Sending verification email...";
    status.className   = "text-center text-sm mb-4 text-yellow-400";

    try {
        const result = await Api.enrollUser(email, password);

        if (result.success && result.verification_sent) {
            currentUsername = email;
            currentPassword = password;

            // Swap to the "check your email" screen and start polling.
            document.getElementById("username-section").classList.add("hidden");
            document.getElementById("verify-email-address").textContent = email;
            document.getElementById("verify-email-section").classList.remove("hidden");
            startVerificationPolling(email);
        } else if (result.detail) {
            status.textContent = "❌ " + result.detail;
            status.className   = "text-center text-sm mb-4 text-red-400";
        } else {
            status.textContent = "❌ Failed to send verification email.";
            status.className   = "text-center text-sm mb-4 text-red-400";
        }
    } catch (err) {
        status.textContent = "❌ Network error.";
        status.className   = "text-center text-sm mb-4 text-red-400";
        console.error("Account creation error:", err);
    }
}

// ── Email verification polling ───────────────────────────────────────────
let _verifyPollTimer = null;
let _verifyPollStart = 0;
const VERIFY_POLL_INTERVAL_MS = 3000;
const VERIFY_POLL_TIMEOUT_MS  = 15 * 60 * 1000;  // matches backend TTL

function startVerificationPolling(email) {
    _verifyPollStart = Date.now();
    stopVerificationPolling();
    _verifyPollTimer = setInterval(() => pollVerification(email), VERIFY_POLL_INTERVAL_MS);
    pollVerification(email);
}

function stopVerificationPolling() {
    if (_verifyPollTimer) {
        clearInterval(_verifyPollTimer);
        _verifyPollTimer = null;
    }
}

async function pollVerification(email) {
    if (Date.now() - _verifyPollStart > VERIFY_POLL_TIMEOUT_MS) {
        stopVerificationPolling();
        const label = document.getElementById("verify-wait-label");
        if (label) { label.textContent = "Link expired — please try again."; label.className = "text-red-400 text-xs ml-2"; }
        return;
    }
    try {
        const res = await Api.checkEmailVerified(email);
        if (res && res.verified) {
            stopVerificationPolling();
            currentPhrase = res.phrase || "biometric voice keystroke authentication";

            KeystrokeCapture.setPhrase(currentPhrase);
            document.querySelectorAll(".phrase-display").forEach(el => {
                el.textContent = currentPhrase;
            });

            document.getElementById("verify-email-section").classList.add("hidden");
            document.getElementById("step-indicator").classList.remove("hidden");
            document.getElementById("step1-dot").querySelector("div")
                .classList.replace("bg-gray-700", "bg-purple-600");

            setTimeout(() => moveToKeystrokeEnrollment(), 400);
        }
    } catch (err) {
        console.warn("verification poll failed:", err);
    }
}

function resendVerification() {
    // "Use a different email" — return user to the account-setup screen.
    stopVerificationPolling();
    document.getElementById("verify-email-section").classList.add("hidden");
    document.getElementById("username-section").classList.remove("hidden");
    const s = document.getElementById("password-status");
    if (s) { s.textContent = ""; s.className = "text-center text-sm mb-4 text-gray-500"; }
}

// ── Step 1: Keystroke Enrollment ──────────────────────────────────────────
function moveToKeystrokeEnrollment() {
    document.getElementById("keystroke-section").classList.remove("hidden");

    document.getElementById("step1-dot").querySelector("div")
        .classList.replace("bg-gray-700", "bg-purple-600");

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

    // ── Real-time quality bar (v3) ───────────────────────────────────────────
    // Fires on every keyup so the user sees quality grow as they type.
    // The bar and badge elements are injected by _ensureQualityBar() below.
    _ensureQualityBar();
    KeystrokeCapture.onQualityUpdate = (score, label) => {
        const bar   = document.getElementById("ks-quality-bar-fill");
        const badge = document.getElementById("ks-quality-badge");
        if (!bar || !badge) return;
        const pct = Math.round(score * 100);
        bar.style.width = pct + "%";
        bar.className   = "h-full rounded-full transition-all duration-200 " + _qualityBarColor(label);
        badge.textContent = _qualityBadgeText(label);
        badge.className   = "text-xs font-medium px-2 py-0.5 rounded " + _qualityBadgeColor(label);
    };

    _updateKeystrokeProgress();

    // Colour feedback on every keystroke + auto-submit on keyup when phrase matches
    let _submitPending = false;
    input.oninput = () => {
        const trimmed = input.value.trim();
        const target  = KeystrokeCapture.targetPhrase;
        if (trimmed.length > 0 && !target.startsWith(trimmed)) {
            input.classList.add("border-red-500");
            input.classList.remove("border-gray-700", "border-green-500");
        } else if (trimmed === target) {
            input.classList.add("border-green-500");
            input.classList.remove("border-gray-700", "border-red-500");
        } else {
            input.classList.remove("border-red-500", "border-green-500");
            input.classList.add("border-gray-700");
        }
    };
    input.onkeyup = () => {
        if (_submitPending) return;
        const trimmed = input.value.trim();
        const target  = KeystrokeCapture.targetPhrase;
        if (trimmed === target) {
            // Wait for this keyup event to fully register in KeystrokeCapture
            // before submitting — fast typists can complete the phrase mid-keystroke,
            // cutting off the final key's release timing data.
            // 120ms is enough for the keyup timestamp to land without feeling sluggish.
            _submitPending = true;
            input.value    = target; // normalise — remove any trailing space
            setTimeout(() => {
                input.disabled = true;
                submitKeystroke();
            }, 120);
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
    const btn    = document.getElementById("keystroke-submit-btn");

    // Trim before validating — prevents trailing space from blocking submission
    if (!KeystrokeCapture.validatePhrase(input.value.trim())) {
        status.textContent = "❌ Phrase doesn't match — please type exactly as shown.";
        status.className = "text-center text-sm mb-2 text-red-400";
        _resetKeystrokeInput();
        return;
    }

    if (btn) { btn.disabled = true; btn.textContent = "⏳ Saving…"; }

    const features = KeystrokeCapture.extractFeatures();
    if (!features) {
        status.textContent = "❌ Not enough keystroke data — try again.";
        status.className = "text-center text-sm mb-2 text-red-400";
        _resetKeystrokeInput();
        return;
    }

    // ── Quality gate (v3) ────────────────────────────────────────────────────
    // Reject samples below the acceptable threshold before sending to the
    // backend.  This mirrors TypingDNA's quality gate and prevents distracted
    // or interrupted typing from contaminating the enrollment model.
    const { score: qualityScore, label: qualityLabel, details: qualityDetails } =
        KeystrokeCapture.getQuality();

    if (qualityScore < 0.30) {
        const hint = qualityDetails.rhythmCv > 0.65
            ? "Your pace varied a lot — try to type steadily from start to finish."
            : qualityDetails.backRatio > 0.15
            ? "Too many corrections — type carefully but don't stop."
            : qualityDetails.dwellMean < 25
            ? "Typing too fast — slow down slightly."
            : "Try again at your natural pace.";
        status.textContent = `⚠️ Low sample quality — ${hint}`;
        status.className = "text-center text-sm mb-2 text-yellow-400";
        if (btn) { btn.disabled = false; btn.textContent = "Submit Attempt"; btn.classList.remove("hidden"); }
        _resetKeystrokeInput();
        return;
    }

    console.log(`[Enroll] Quality gate passed: ${(qualityScore * 100).toFixed(0)}% (${qualityLabel})`);
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
            // FIX: shift_lag fields were missing from enrollment — the model
            // uses shift_lag_norm but it was never stored during enrollment,
            // so all enrolled templates had shift_lag_norm = 0.
            shift_lag_mean:          features.shift_lag_mean  || 0,
            shift_lag_std:           features.shift_lag_std   || 0,
            shift_lag_count:         features.shift_lag_count || 0,
            dwell_mean_norm:         features.dwell_mean_norm  || 0,
            dwell_std_norm:          features.dwell_std_norm   || 0,
            flight_mean_norm:        features.flight_mean_norm || 0,
            flight_std_norm:         features.flight_std_norm  || 0,
            p2p_std_norm:            features.p2p_std_norm     || 0,
            r2r_mean_norm:           features.r2r_mean_norm    || 0,
            shift_lag_norm:          features.shift_lag_norm   || 0,
            extra_digraphs:          features.extra_digraphs   || {},
            key_dwell_map:           features.key_dwell_map       || {},
            digraph_dd_map:          features.digraph_dd_map      || {},
            digraph_du_map:          features.digraph_du_map      || {},
            digraph_ud_map:          features.digraph_ud_map      || {},
            digraph_uu_map:          features.digraph_uu_map      || {},
            flight_per_digraph:      features.flight_per_digraph  || {},
            trigraph_map:            features.trigraph_map        || {},
        });

        if (!result.success) {
            status.textContent = "❌ " + (result.detail || "Save failed — please try again.");
            status.className = "text-center text-sm mb-2 text-red-400";
            // Show submit button so user knows they need to retype
            if (btn) { btn.disabled = false; btn.textContent = "Submit Attempt"; btn.classList.remove("hidden"); }
            _resetKeystrokeInput();
            return;
        }

        // ── Quality feedback ─────────────────────────────────────────────
        attemptSpeeds.push(features.typing_speed_cpm);

        // v3: feed this attempt into the cross-attempt consistency tracker
        KeystrokeCapture.addPreviousAttempt(features);

        let qualityNote = "";
        // Use the quality gate score we already computed for richer feedback
        if (qualityLabel === 'strong') {
            qualityNote = "Excellent pattern — very consistent!";
        } else if (qualityLabel === 'good') {
            qualityNote = qualityDetails.rhythm > qualityDetails.dwell
                ? "Good rhythm. Stay consistent."
                : "Good sample. Try to keep the same pace.";
        } else {
            qualityNote = qualityDetails.rhythmCv > 0.50
                ? "Pace was a bit uneven — try to type more steadily."
                : "Try to keep the same speed each time.";
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

// ── Step 2: Voice Enrollment ──────────────────────────────────────────────
function moveToVoiceEnrollment() {
    document.getElementById("keystroke-section").classList.add("hidden");
    document.getElementById("voice-section").classList.remove("hidden");

    document.getElementById("step2-dot").querySelector("div")
        .classList.replace("bg-gray-700", "bg-purple-600");
    document.getElementById("progress-line").style.width = "100%";

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

        // ── Consistency check failed — sample rejected, ask user to redo ──
        if (result.consistency_warning) {
            status.textContent = "⚠️ " + result.message;
            status.className   = "text-center text-sm mb-4 text-yellow-400";

            const btn = document.getElementById("record-btn");
            if (btn) {
                btn.disabled    = false;
                btn.textContent = `🎤 Re-record ${voiceAttemptsSaved + 1}/${VOICE_TARGET}`;
            }
            return;  // counter unchanged — user records the same slot again
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

// ── Step 3: Security Question ─────────────────────────────────────────────
function moveToSecurityQuestion() {
    document.getElementById("voice-section").classList.add("hidden");
    document.getElementById("security-section").classList.remove("hidden");

    document.getElementById("step3-dot").querySelector("div")
        .classList.replace("bg-gray-700", "bg-purple-600");
    document.getElementById("progress-line2").style.width = "100%";

    saveEnrollState("security");
}

async function submitSecurityQuestion() {
    const question = document.getElementById("security-question-select").value;
    const answer   = document.getElementById("security-answer").value.trim();
    const btn      = document.querySelector("#security-section button");
    const status   = document.getElementById("security-status");

    if (!question || !answer) {
        if (status) { status.textContent = "❌ Please select a question and provide an answer."; status.className = "text-center text-sm mb-4 text-red-400"; }
        else alert("Please select a question and provide an answer.");
        return;
    }

    // Guard against double-clicks while request is in flight
    if (btn.disabled) return;
    btn.disabled    = true;
    btn.textContent = "⏳ Saving…";
    if (status) { status.textContent = "⏳ Saving your answer…"; status.className = "text-center text-sm mb-4 text-yellow-400"; }

    try {
        const result = await Api.enrollSecurity(currentUsername, question, answer);
        if (result.success) {
            clearEnrollState();
            document.getElementById("security-section").classList.add("hidden");
            document.getElementById("success-section").classList.remove("hidden");
        } else {
            if (status) { status.textContent = "❌ " + (result.detail || "Unknown error"); status.className = "text-center text-sm mb-4 text-red-400"; }
            else alert("Error saving security question: " + (result.detail || "Unknown error"));
            btn.disabled    = false;
            btn.textContent = "Complete Enrollment ✅";
        }
    } catch (err) {
        if (status) { status.textContent = "❌ Could not connect to server."; status.className = "text-center text-sm mb-4 text-red-400"; }
        else alert("Could not connect to server.");
        btn.disabled    = false;
        btn.textContent = "Complete Enrollment ✅";
        console.error(err);
    }
}
// ── Quality bar helpers (v3) ──────────────────────────────────────────────────
// These inject and style the real-time quality bar that wires to
// KeystrokeCapture.onQualityUpdate.

function _ensureQualityBar() {
    if (document.getElementById("ks-quality-bar")) return;

    // Find the keystroke status element and inject bar beneath it
    const status = document.getElementById("keystroke-status");
    if (!status) return;

    const wrapper = document.createElement("div");
    wrapper.id        = "ks-quality-bar";
    wrapper.className = "mt-2 px-1";
    wrapper.innerHTML = `
        <div class="flex items-center justify-between mb-1">
            <span class="text-xs text-gray-500">Sample quality</span>
            <span id="ks-quality-badge" class="text-xs font-medium px-2 py-0.5 rounded bg-gray-700 text-gray-400">—</span>
        </div>
        <div class="h-1.5 w-full rounded-full bg-gray-700 overflow-hidden">
            <div id="ks-quality-bar-fill" class="h-full rounded-full transition-all duration-200 bg-gray-600" style="width:0%"></div>
        </div>`;
    status.insertAdjacentElement("afterend", wrapper);
}

function _qualityBarColor(label) {
    return {
        strong:     "bg-green-500",
        good:       "bg-blue-500",
        acceptable: "bg-yellow-500",
        weak:       "bg-red-500",
    }[label] || "bg-gray-600";
}

function _qualityBadgeText(label) {
    return { strong: "Strong", good: "Good", acceptable: "Acceptable", weak: "Weak" }[label] || "—";
}

function _qualityBadgeColor(label) {
    return {
        strong:     "bg-green-900 text-green-300",
        good:       "bg-blue-900 text-blue-300",
        acceptable: "bg-yellow-900 text-yellow-300",
        weak:       "bg-red-900 text-red-300",
    }[label] || "bg-gray-700 text-gray-400";
}