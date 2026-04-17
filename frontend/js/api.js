// frontend/js/api.js
// FIXED v4:
//  1. API_BASE auto-detects the current host — works for both localhost AND ngrok/remote.
//     No longer relies on main.py rewriting this file at serve time.
//  2. enrollVoice and verifyVoice forward ALL 62 features
//  3. verifyVoice passes snr_db for server-side logging

// Auto-detect the backend URL from wherever the page is being served.
// - On localhost: resolves to "http://127.0.0.1:8000/api"
// - On ngrok:     resolves to "https://abc123.ngrok-free.app/api"
// - On any host:  always correct, no manual changes needed
const API_BASE = `${window.location.protocol}//${window.location.host}/api`;

// ── Shared voice feature builder ──────────────────────────────────────────────
// Single source of truth — used by both enroll and verify so they can never drift.
function _buildVoicePayload(username, d) {
    return {
        username,
        mfcc_features:          d.mfcc_features          || [],
        mfcc_std:               d.mfcc_std               || [],
        delta_mfcc_mean:        d.delta_mfcc_mean        || [],   // v2 feature
        delta2_mfcc_mean:       d.delta2_mfcc_mean       || [],   // v2 feature
        pitch_mean:             d.pitch_mean             || 0,
        pitch_std:              d.pitch_std              || 0,
        speaking_rate:          d.speaking_rate          || 0,
        energy_mean:            d.energy_mean            || 0,
        energy_std:             d.energy_std             || 0,
        zcr_mean:               d.zcr_mean               || 0,
        spectral_centroid_mean: d.spectral_centroid_mean || 0,
        spectral_rolloff_mean:  d.spectral_rolloff_mean  || 0,
        spectral_flux_mean:     d.spectral_flux_mean     || 0,    // v2 feature
        voiced_fraction:        d.voiced_fraction        || 0,    // v2 feature
        snr_db:                 d.snr_db                 || 0,    // for logging
        // ECAPA-TDNN embedding (kept for compatibility).
        ecapa_embedding:        d.ecapa_embedding        || [],
        // Raw WAV audio as base64 — forwarded to Azure Speaker Recognition for auth.
        raw_audio_b64:          d.raw_audio_b64          || "",
        // Whisper transcript — needed for phrase verification at /auth/voice.
        transcript:             d.transcript             || "",
    };
}

const Api = {

    // ── Enrollment ────────────────────────────────────────────────────────

    async enrollUser(username, password) {
        const res = await fetch(`${API_BASE}/enroll/user`, {
            method:  "POST",
            headers: { "Content-Type": "application/json" },
            body:    JSON.stringify({ username, password })
        });
        return res.json();
    },

    async enrollKeystroke(username, features) {
        const res = await fetch(`${API_BASE}/enroll/keystroke`, {
            method:  "POST",
            headers: { "Content-Type": "application/json" },
            body:    JSON.stringify({ username, ...features })
        });
        return res.json();
    },

    async enrollVoice(username, fullFeatureDict) {
        const res = await fetch(`${API_BASE}/enroll/voice`, {
            method:  "POST",
            headers: { "Content-Type": "application/json" },
            body:    JSON.stringify(_buildVoicePayload(username, fullFeatureDict))
        });
        return res.json();
    },

    async enrollSecurity(username, question, answer) {
        const res = await fetch(`${API_BASE}/enroll/security`, {
            method:  "POST",
            headers: { "Content-Type": "application/json" },
            body:    JSON.stringify({ username, question, answer })
        });
        return res.json();
    },

    // ── Authentication ────────────────────────────────────────────────────

    async verifyPassword(username, password) {
        const res = await fetch(`${API_BASE}/auth/password`, {
            method:  "POST",
            headers: { "Content-Type": "application/json" },
            body:    JSON.stringify({ username, password })
        });
        return res.json();
    },

    async verifyKeystroke(username, features) {
        const res = await fetch(`${API_BASE}/auth/keystroke`, {
            method:  "POST",
            headers: { "Content-Type": "application/json" },
            body:    JSON.stringify({ username, ...features })
        });
        return res.json();
    },

    async verifyVoice(username, fullFeatureDict) {
        const res = await fetch(`${API_BASE}/auth/voice`, {
            method:  "POST",
            headers: { "Content-Type": "application/json" },
            body:    JSON.stringify(_buildVoicePayload(username, fullFeatureDict))
        });
        return res.json();
    },

    async getPhrase(email) {
        const res = await fetch(`${API_BASE}/auth/phrase/${encodeURIComponent(email)}`);
        return res.json();
    },

    async getSecurityQuestion(username, password) {
        const res = await fetch(`${API_BASE}/auth/security-question`, {
            method:  "POST",
            headers: { "Content-Type": "application/json" },
            body:    JSON.stringify({ username, password })
        });
        return res.json();
    },

    async verifySecurityQuestion(username, answer) {
        const res = await fetch(`${API_BASE}/auth/security`, {
            method:  "POST",
            headers: { "Content-Type": "application/json" },
            body:    JSON.stringify({ username, answer })
        });
        return res.json();
    },

    async fuseScores(username, keystrokeScore, voiceScore, keystrokePassed, voicePassed) {
        const res = await fetch(`${API_BASE}/auth/fuse`, {
            method:  "POST",
            headers: { "Content-Type": "application/json" },
            body:    JSON.stringify({
                username,
                keystroke_score:  keystrokeScore,
                voice_score:      voiceScore,
                keystroke_passed: keystrokePassed,
                voice_passed:     voicePassed,
            })
        });
        return res.json();
    }
};