// frontend/js/keystroke.js
// Keystroke Dynamics Capture — Fixed & Improved v2
// Fixes applied:
//   1. Repeated-key dwell bug (ordered pairing instead of code map)
//   2. Dead digraphs replaced with digraphs from actual phrase
//   3. Spacebar added to hand detection (thumb) for word-boundary alternation
//   4. Shift-key lag tracking (new biometric feature)
//   5. Inter-session normalization (speed-independent ratio features)
//   6. Pre-auth quality filter before hitting the ML backend

const KeystrokeCapture = {
    events: [],
    isCapturing: false,
    startTime: null,
    endTime: null,
    textBuffer: [],
    backspaceCount: 0,
    shiftPressTime: null,
    shiftKeyLags: [],
    targetPhrase: "Biometric Voice Keystroke Authentication",

    // ── Keyboard layout ───────────────────────────────────────────────────
    // FIX: spacebar now assigned to 'thumb' so word-boundary hand
    //      alternation is counted correctly (was silently skipped before)
    keyboardLayout: {
        left_hand:    new Set('qwertasdfgzxcvb12345'),
        right_hand:   new Set('yuiophjklnm67890'),
        thumb:        new Set([' ']),
        left_pinky:   new Set('qa1z'),
        left_ring:    new Set('wsx2'),
        left_middle:  new Set('edc3'),
        left_index:   new Set('rtfgvb45'),
        right_index:  new Set('yhnujm67'),
        right_middle: new Set('ik8'),
        right_ring:   new Set('ol9'),
        right_pinky:  new Set('p0'),
    },

    // FIX: digraphs now extracted from the ACTUAL phrase
    // "Biometric Voice Keystroke Authentication"
    // Previous list had 6/8 digraphs never appearing → always 0 → model noise
    trackedDigraphs: [
        'bi','io','om','me','et','tr','ri','ic',   // Biometric
        'vo','oi','ce',                             // Voice
        'ke','ey','ys','st','ro','ok',              // Keystroke
        'au','ut','th','he','en','nt','ti','ca','at','on'  // Authentication
    ],

    // ── Attach to input element ───────────────────────────────────────────
    attach(inputElementId) {
        const input = document.getElementById(inputElementId);
        if (!input) {
            console.error(`[KeystrokeCapture] Element #${inputElementId} not found`);
            return;
        }
        this.reset();
        this.isCapturing = true;
        input.addEventListener('keydown', (e) => this.onKeyDown(e));
        input.addEventListener('keyup',   (e) => this.onKeyUp(e));
        console.log('[KeystrokeCapture] Attached to:', inputElementId);
    },

    // ── Event handlers ────────────────────────────────────────────────────
    onKeyDown(e) {
        if (!this.isCapturing) return;
        const now = performance.now();

        if (this.startTime === null) this.startTime = now;

        // NEW: track shift press timestamp for shift-lag feature
        if (e.code === 'ShiftLeft' || e.code === 'ShiftRight') {
            this.shiftPressTime = now;
        }

        // NEW: record lag between Shift press and the letter key
        // Phrase has 4 capital letters → 4 shift-lag samples per attempt
        if (e.shiftKey && e.key.length === 1 && this.shiftPressTime !== null) {
            const lag = now - this.shiftPressTime;
            if (lag >= 0 && lag < 500) {  // sanity: ignore accidental holds > 500ms
                this.shiftKeyLags.push(lag);
            }
        }

        // FIX: store 'used' flag so repeated keys pair correctly in dwell calc
        this.events.push({
            type: 'press',
            key:  e.key,
            code: e.code,
            time: now,
            used: false,
        });

        if (e.key === 'Backspace') {
            this.backspaceCount++;
            if (this.textBuffer.length > 0) this.textBuffer.pop();
        } else if (e.key.length === 1) {
            this.textBuffer.push(e.key);
        }
    },

    onKeyUp(e) {
        if (!this.isCapturing) return;
        const now = performance.now();
        this.endTime = now;
        this.events.push({
            type: 'release',
            key:  e.key,
            code: e.code,
            time: now,
            used: false,
        });
    },

    // ── Hand / finger helpers ─────────────────────────────────────────────
    getHand(char) {
        if (char === ' ') return 'thumb';
        char = char.toLowerCase();
        if (this.keyboardLayout.left_hand.has(char))  return 'left';
        if (this.keyboardLayout.right_hand.has(char)) return 'right';
        return null;
    },

    getFinger(char) {
        if (char === ' ') return 'thumb';
        char = char.toLowerCase();
        for (const [finger, keys] of Object.entries(this.keyboardLayout)) {
            if (keys.has && keys.has(char)) return finger;
        }
        return null;
    },

    // ── FIX: dwell pairing by chronological order, not by code map ────────
    // Old approach: pressTimeMap[e.code] = e.time
    //   → second 'e' in "Biometric" overwrites the first, wrong dwell for both
    // New approach: for each press, find the earliest unused release with the
    //   same code that comes AFTER this press timestamp
    _buildDwellPairs() {
        const pressEvents   = this.events.filter(e => e.type === 'press');
        const releaseEvents = this.events.filter(e => e.type === 'release')
                                         .map(e => ({ ...e, used: false }));

        const dwellTimes  = [];
        const pairedPress = [];

        for (const press of pressEvents) {
            const matchIdx = releaseEvents.findIndex(
                r => !r.used && r.code === press.code && r.time > press.time
            );
            if (matchIdx !== -1) {
                const release = releaseEvents[matchIdx];
                release.used  = true;
                const dwell   = release.time - press.time;
                // Sanity bounds: ignore holds < 0ms or > 2s (accidental holds)
                if (dwell >= 0 && dwell < 2000) {
                    dwellTimes.push(dwell);
                    pairedPress.push(press);
                }
            }
        }
        return { dwellTimes, pairedPress };
    },

    // ── Main feature extraction ───────────────────────────────────────────
    extractFeatures() {
        const pressEvents   = this.events.filter(e => e.type === 'press');
        const releaseEvents = this.events.filter(e => e.type === 'release');

        if (pressEvents.length < 5) {
            console.warn('[KeystrokeCapture] Too few press events:', pressEvents.length);
            return null;
        }

        // Dwell times (fixed)
        const { dwellTimes } = this._buildDwellPairs();

        // Flight times: release[i] → press[i+1]
        const flightTimes = [];
        for (let i = 0; i < releaseEvents.length - 1; i++) {
            if (i < pressEvents.length - 1) {
                const flight = pressEvents[i + 1].time - releaseEvents[i].time;
                // Allow slight overlap (fast typists can press next key before releasing)
                if (flight > -50 && flight < 2000) {
                    flightTimes.push(Math.max(0, flight));
                }
            }
        }

        // Press-to-press
        const p2pTimes = [];
        for (let i = 0; i < pressEvents.length - 1; i++) {
            const dt = pressEvents[i + 1].time - pressEvents[i].time;
            if (dt >= 0 && dt < 3000) p2pTimes.push(dt);
        }

        // Release-to-release
        const r2rTimes = [];
        for (let i = 0; i < releaseEvents.length - 1; i++) {
            const dt = releaseEvents[i + 1].time - releaseEvents[i].time;
            if (dt >= 0 && dt < 3000) r2rTimes.push(dt);
        }

        // Digraphs — only tracked pairs that exist in the actual phrase
        const digraphMap = {};
        this.trackedDigraphs.forEach(dg => { digraphMap[dg] = []; });
        for (let i = 0; i < pressEvents.length - 1; i++) {
            const pair = (pressEvents[i].key + pressEvents[i + 1].key).toLowerCase();
            if (digraphMap[pair] !== undefined) {
                const dt = pressEvents[i + 1].time - pressEvents[i].time;
                if (dt >= 0 && dt < 3000) digraphMap[pair].push(dt);
            }
        }

        // Typing speed
        const duration = (this.endTime && this.startTime)
            ? (this.endTime - this.startTime) / 1000
            : 0;
        const typingSpeedCpm = duration > 0
            ? (this.textBuffer.length / duration) * 60
            : 0;

        // Rhythm
        const rhythmMean = _mean(p2pTimes);
        const rhythmStd  = _std(p2pTimes);
        const rhythmCv   = rhythmMean > 0 ? rhythmStd / rhythmMean : 0;

        // Pauses > 500ms
        const pauses = p2pTimes.filter(t => t > 500);

        // Hand alternation (FIX: space now counted via 'thumb')
        const text = this.textBuffer.join('').toLowerCase();
        let alternations = 0;
        const sameHandSeqs = [];
        let currentSeq = 0;
        for (let i = 0; i < text.length - 1; i++) {
            const currHand = this.getHand(text[i]);
            const nextHand = this.getHand(text[i + 1]);
            if (currHand && nextHand && currHand !== nextHand) {
                alternations++;
                if (currentSeq > 0) sameHandSeqs.push(currentSeq);
                currentSeq = 0;
            } else {
                currentSeq++;
            }
        }
        if (currentSeq > 0) sameHandSeqs.push(currentSeq);

        // Finger transitions
        let fingerTransitions = 0;
        for (let i = 0; i < text.length - 1; i++) {
            const f1 = this.getFinger(text[i]);
            const f2 = this.getFinger(text[i + 1]);
            if (f1 && f2 && f1 !== f2) fingerTransitions++;
        }

        // Seek times > 300ms
        const seekTimes = p2pTimes.filter(t => t > 300);

        // Backspace
        const totalKeys      = pressEvents.length;
        const backspaceRatio = totalKeys > 0 ? this.backspaceCount / totalKeys : 0;

        // Build raw feature object
        const raw = {
            // Core timing
            dwell_mean:    _mean(dwellTimes),
            dwell_std:     _std(dwellTimes),
            dwell_median:  _median(dwellTimes),
            dwell_min:     dwellTimes.length ? Math.min(...dwellTimes) : 0,
            dwell_max:     dwellTimes.length ? Math.max(...dwellTimes) : 0,

            flight_mean:   _mean(flightTimes),
            flight_std:    _std(flightTimes),
            flight_median: _median(flightTimes),

            p2p_mean:      _mean(p2pTimes),
            p2p_std:       _std(p2pTimes),

            r2r_mean:      _mean(r2rTimes),
            r2r_std:       _std(r2rTimes),

            // Digraphs (all now present in phrase — no more zero features)
            ...Object.fromEntries(
                this.trackedDigraphs.map(dg => [
                    `digraph_${dg}`, _mean(digraphMap[dg])
                ])
            ),

            // Behavioral
            typing_speed_cpm:         typingSpeedCpm,
            typing_duration:          duration,
            rhythm_mean:              rhythmMean,
            rhythm_std:               rhythmStd,
            rhythm_cv:                rhythmCv,
            pause_count:              pauses.length,
            pause_mean:               _mean(pauses),

            backspace_ratio:          backspaceRatio,
            backspace_count:          this.backspaceCount,
            hand_alternation_ratio:   text.length > 1
                                      ? alternations / (text.length - 1) : 0,
            same_hand_sequence_mean:  _mean(sameHandSeqs),
            finger_transition_ratio:  text.length > 1
                                      ? fingerTransitions / (text.length - 1) : 0,
            seek_time_mean:           _mean(seekTimes),
            seek_time_count:          seekTimes.length,

            // NEW: shift-lag features
            // Captures the time between pressing Shift and the letter key
            // for each capitalized word — strong consistent biometric signal
            shift_lag_mean:  _mean(this.shiftKeyLags),
            shift_lag_std:   _std(this.shiftKeyLags),
            shift_lag_count: this.shiftKeyLags.length,

            // Raw arrays for database storage
            dwell_times:  dwellTimes,
            flight_times: flightTimes,
            typing_speed: typingSpeedCpm / 60,
        };

        // Apply inter-session normalization and return
        const features = this._normalize(raw);
        console.log(`[KeystrokeCapture] Features extracted: ${Object.keys(features).length}`);
        return features;
    },

    // ── Inter-session normalization ───────────────────────────────────────
    // Raw ms values shift between sessions (fatigue, mood, device).
    // Normalizing against p2p_mean makes features speed-independent so the
    // model learns rhythm pattern, not absolute timing → reduces FFR.
    _normalize(f) {
        const baseline = f.p2p_mean;
        if (baseline <= 0) return f;

        return {
            ...f,
            dwell_mean_norm:  f.dwell_mean    / baseline,
            dwell_std_norm:   f.dwell_std     / baseline,
            flight_mean_norm: f.flight_mean   / baseline,
            flight_std_norm:  f.flight_std    / baseline,
            p2p_std_norm:     f.p2p_std       / baseline,
            r2r_mean_norm:    f.r2r_mean      / baseline,
            shift_lag_norm:   f.shift_lag_mean / baseline,
            // rhythm_cv is already normalized (std/mean) — keep as-is
        };
    },

    // ── Pre-auth quality check ────────────────────────────────────────────
    // Call this BEFORE sending features to the ML backend.
    // If quality is 'low', show the user a "please retype" message instead
    // of wasting a prediction on a bad sample — directly reduces FFR.
    getTypingQuality(features) {
        if (!features)
            return { quality: 'low', reason: 'no features extracted' };

        if (features.rhythm_cv > 0.8)
            return { quality: 'low', reason: 'inconsistent rhythm — please retype' };

        if (features.typing_speed_cpm < 15 || features.typing_speed_cpm > 700)
            return { quality: 'low', reason: 'abnormal typing speed — please retype' };

        if (features.backspace_ratio > 0.3)
            return { quality: 'low', reason: 'too many corrections — please retype' };

        if (features.typing_duration < 2 || features.typing_duration > 30)
            return { quality: 'low', reason: 'unusual duration — please retype' };

        return { quality: 'high', reason: 'ok' };
    },

    validatePhrase(inputValue) {
        return inputValue.trim() === this.targetPhrase;
    },

    reset() {
        this.events         = [];
        this.isCapturing    = false;
        this.startTime      = null;
        this.endTime        = null;
        this.textBuffer     = [];
        this.backspaceCount = 0;
        this.shiftPressTime = null;
        this.shiftKeyLags   = [];
    },
};

// ── Math helpers (prefixed with _ to avoid global namespace conflicts) ────────
function _mean(arr) {
    if (!arr || arr.length === 0) return 0;
    return arr.reduce((a, b) => a + b, 0) / arr.length;
}

function _std(arr) {
    if (!arr || arr.length < 2) return 0;
    const m = _mean(arr);
    return Math.sqrt(arr.reduce((a, b) => a + (b - m) ** 2, 0) / arr.length);
}

function _median(arr) {
    if (!arr || arr.length === 0) return 0;
    const sorted = [...arr].sort((a, b) => a - b);
    const mid    = Math.floor(sorted.length / 2);
    return sorted.length % 2 !== 0
        ? sorted[mid]
        : (sorted[mid - 1] + sorted[mid]) / 2;
}