// frontend/js/api.js

const API_BASE = "http://127.0.0.1:8000/api";

const Api = {

    // ── Enrollment ────────────────────────────────────────────────────────

    async enrollUser(username) {
        const res = await fetch(`${API_BASE}/enroll/user`, {
            method:  "POST",
            headers: { "Content-Type": "application/json" },
            body:    JSON.stringify({ username })
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

    async enrollKeystroke(username, features) {
        const res = await fetch(`${API_BASE}/enroll/keystroke`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ username, ...features })
        });
        return res.json();
    },

    async enrollVoice(username, fullFeatureDict) {
        const res = await fetch(`${API_BASE}/enroll/voice`, {
            method:  "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                username,
                mfcc_features:          fullFeatureDict.mfcc_features          || [],
                mfcc_std:               fullFeatureDict.mfcc_std               || [],
                pitch_mean:             fullFeatureDict.pitch_mean             || 0,
                pitch_std:              fullFeatureDict.pitch_std              || 0,
                speaking_rate:          fullFeatureDict.speaking_rate          || 0,
                energy_mean:            fullFeatureDict.energy_mean            || 0,
                energy_std:             fullFeatureDict.energy_std             || 0,
                zcr_mean:               fullFeatureDict.zcr_mean               || 0,
                spectral_centroid_mean: fullFeatureDict.spectral_centroid_mean || 0,
                spectral_rolloff_mean:  fullFeatureDict.spectral_rolloff_mean  || 0,
            })
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

    // ── Authentication ────────────────────────────────────
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
            body: JSON.stringify({
                username,
                mfcc_features:          fullFeatureDict.mfcc_features          || [],
                mfcc_std:               fullFeatureDict.mfcc_std               || [],
                pitch_mean:             fullFeatureDict.pitch_mean             || 0,
                pitch_std:              fullFeatureDict.pitch_std              || 0,
                speaking_rate:          fullFeatureDict.speaking_rate          || 0,
                energy_mean:            fullFeatureDict.energy_mean            || 0,
                energy_std:             fullFeatureDict.energy_std             || 0,
                zcr_mean:               fullFeatureDict.zcr_mean               || 0,
                spectral_centroid_mean: fullFeatureDict.spectral_centroid_mean || 0,
                spectral_rolloff_mean:  fullFeatureDict.spectral_rolloff_mean  || 0,
            })
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