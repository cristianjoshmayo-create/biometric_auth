// frontend/js/api.js

const API_BASE = "http://127.0.0.1:8000/api";

const Api = {

    // ── Enrollment ────────────────────────────────────────
    async enrollUser(username) {
        const res = await fetch(`${API_BASE}/enroll/user`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ username })
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

    async enrollVoice(username, mfccFeatures) {
        const res = await fetch(`${API_BASE}/enroll/voice`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ username, mfcc_features: mfccFeatures })
        });
        return res.json();
    },

    async enrollSecurity(username, question, answer) {
        const res = await fetch(`${API_BASE}/enroll/security`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ username, question, answer })
        });
        return res.json();
    },

    // ── Authentication ────────────────────────────────────
    async verifyKeystroke(username, features) {
        const res = await fetch(`${API_BASE}/auth/keystroke`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ username, ...features })
        });
        return res.json();
    },

    async verifyVoice(username, mfccFeatures) {
        const res = await fetch(`${API_BASE}/auth/voice`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ username, mfcc_features: mfccFeatures })
        });
        return res.json();
    },

    async getSecurityQuestion(username) {
        const res = await fetch(`${API_BASE}/auth/security-question/${username}`);
        return res.json();
    },

    async verifySecurityQuestion(username, answer) {
        const res = await fetch(`${API_BASE}/auth/security`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ username, answer })
        });
        return res.json();
    }
};