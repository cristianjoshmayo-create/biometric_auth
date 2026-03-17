// frontend/js/api.js
// FIXED v3:
//  1. Single API_BASE constant — change here, affects everywhere (was duplicated in speech.js too)
//  2. enrollVoice and verifyVoice now forward ALL 62 features (delta_mfcc_mean,
//     delta2_mfcc_mean, spectral_flux_mean, voiced_fraction were silently dropped)
//  3. verifyVoice passes snr_db for server-side logging

const API_BASE = "http://127.0.0.1:8000/api";

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

    async getSecurityQuestion(username) {
        const res = await fetch(`${API_BASE}/auth/security-question/${username}`);
        return res.json();
    },

    async verifySecurityQuestion(username, answer) {
        const res = await fetch(`${API_BASE}/auth/security`, {
            method:  "POST",
            headers: { "Content-Type": "application/json" },
            body:    JSON.stringify({ username, answer })
        });
        return res.json();
    }
};