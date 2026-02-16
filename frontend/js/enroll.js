// frontend/js/enroll.js
// Controls the enrollment flow across all 3 steps

let currentUsername = "";
let keystrokeAttempts = [];
let voiceAttempts = [];
let currentKeystrokeAttempt = 1;
let currentVoiceAttempt = 1;

// ── Step 0: Start enrollment ──────────────────────────────
function startEnrollment() {
    const username = document.getElementById("username-input").value.trim();
    if (!username) {
        alert("Please enter a username first.");
        return;
    }

    currentUsername = username;

    // Hide username section, show step indicator + keystroke section
    document.getElementById("username-section").classList.add("hidden");
    document.getElementById("step-indicator").classList.remove("hidden");
    document.getElementById("keystroke-section").classList.remove("hidden");

    // Attach keystroke capture to input
    KeystrokeCapture.attach("keystroke-input");
    document.getElementById("keystroke-status").textContent = "Start typing when ready";
}

// ── Step 1: Keystroke Enrollment ─────────────────────────
function submitKeystroke() {
    const input = document.getElementById("keystroke-input");
    const status = document.getElementById("keystroke-status");

    // Validate phrase
    if (!KeystrokeCapture.validatePhrase(input.value)) {
        status.textContent = "❌ Phrase doesn't match. Please type exactly as shown.";
        status.classList.add("text-red-400");
        input.value = "";
        KeystrokeCapture.reset();
        KeystrokeCapture.attach("keystroke-input");
        return;
    }

    // Extract features
    const features = KeystrokeCapture.extractFeatures();
    if (!features || features.totalKeys < 5) {
        status.textContent = "❌ Not enough keystroke data captured. Try again.";
        return;
    }

    keystrokeAttempts.push(features);

    // Update attempt dots
    document.getElementById(`attempt-${currentKeystrokeAttempt}`)
        .classList.replace("bg-gray-700", "bg-purple-600");

    status.textContent = `✅ Attempt ${currentKeystrokeAttempt} recorded!`;
    status.classList.remove("text-red-400");
    status.classList.add("text-green-400");

    currentKeystrokeAttempt++;

    if (currentKeystrokeAttempt <= 3) {
        // More attempts needed
        document.getElementById("attempt-label").textContent =
            `Attempt ${currentKeystrokeAttempt} of 3`;
        input.value = "";
        KeystrokeCapture.reset();
        KeystrokeCapture.attach("keystroke-input");

        setTimeout(() => {
            status.textContent = "Start typing when ready";
            status.classList.remove("text-green-400");
        }, 1000);

    } else {
        // All 3 attempts done — save to backend
        saveKeystrokeEnrollment();
    }
}

async function saveKeystrokeEnrollment() {
    const status = document.getElementById("keystroke-status");
    status.textContent = "💾 Saving keystroke profile...";

    // Average features across 3 attempts
    const avgDwell = averageArrays(keystrokeAttempts.map(a => a.dwellTimes));
    const avgFlight = averageArrays(keystrokeAttempts.map(a => a.flightTimes));
    const avgSpeed = keystrokeAttempts.reduce((s, a) => s + a.typingSpeed, 0)
                     / keystrokeAttempts.length;

    try {
        // First create the user
        await Api.enrollUser(currentUsername);

        // Then save keystroke template
        const result = await Api.enrollKeystroke(currentUsername, {
            dwell_times: avgDwell,
            flight_times: avgFlight,
            typing_speed: avgSpeed
        });

        if (result.success) {
            status.textContent = "✅ Keystroke profile saved!";
            setTimeout(() => moveToVoiceEnrollment(), 1000);
        } else {
            status.textContent = "❌ Error: " + (result.detail || "Unknown error");
        }

    } catch (err) {
        status.textContent = "❌ Could not connect to server.";
        console.error(err);
    }
}

// ── Step 2: Voice Enrollment ──────────────────────────────
function moveToVoiceEnrollment() {
    document.getElementById("keystroke-section").classList.add("hidden");
    document.getElementById("voice-section").classList.remove("hidden");

    // Update step indicator
    document.getElementById("step2-dot").querySelector("div")
        .classList.replace("bg-gray-700", "bg-purple-600");
    document.getElementById("progress-line").style.width = "100%";
}

// Voice recording handled in speech.js
// This is the callback when a recording is done
function onVoiceRecorded(mfccFeatures) {
    voiceAttempts.push(mfccFeatures);

    document.getElementById(`vattempt-${currentVoiceAttempt}`)
        .classList.replace("bg-gray-700", "bg-purple-600");

    currentVoiceAttempt++;

    if (currentVoiceAttempt <= 3) {
        document.getElementById("vattempt-label").textContent =
            `Recording ${currentVoiceAttempt} of 3`;
        document.getElementById("voice-status").textContent =
            `✅ Recording ${currentVoiceAttempt - 1} saved. Click record for next.`;
    } else {
        saveVoiceEnrollment();
    }
}

async function saveVoiceEnrollment() {
    const status = document.getElementById("voice-status");
    status.textContent = "💾 Saving voice profile...";

    const avgMfcc = averageArrays(voiceAttempts);

    try {
        const result = await Api.enrollVoice(currentUsername, avgMfcc);

        if (result.success) {
            status.textContent = "✅ Voice profile saved!";
            setTimeout(() => moveToSecurityQuestion(), 1000);
        } else {
            status.textContent = "❌ Error: " + (result.detail || "Unknown error");
        }
    } catch (err) {
        status.textContent = "❌ Could not connect to server.";
        console.error(err);
    }
}

// ── Step 3: Security Question ─────────────────────────────
function moveToSecurityQuestion() {
    document.getElementById("voice-section").classList.add("hidden");
    document.getElementById("security-section").classList.remove("hidden");

    document.getElementById("step3-dot").querySelector("div")
        .classList.replace("bg-gray-700", "bg-purple-600");
    document.getElementById("progress-line2").style.width = "100%";
}

async function submitSecurityQuestion() {
    const question = document.getElementById("security-question-select").value;
    const answer = document.getElementById("security-answer").value.trim();

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

// ── Helper: Average multiple arrays ──────────────────────
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