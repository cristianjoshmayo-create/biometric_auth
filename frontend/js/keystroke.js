// frontend/js/keystroke.js
// Keystroke Dynamics Capture — Fixed & Improved v2

const KeystrokeCapture = {
    events: [],
    isCapturing: false,
    startTime: null,
    endTime: null,
    textBuffer: [],
    backspaceCount: 0,
    targetPhrase: "biometric voice keystroke authentication",

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

    trackedDigraphs: [
        'bi','io','om','me','et','tr','ri','ic',
        'vo','oi','ce',
        'ke','ey','ys','st','ro','ok',
        'au','ut','th','he','en','nt','ti','ca','at','on'
    ],

    attach(inputElementId) {
        const input = document.getElementById(inputElementId);
        if (!input) {
            console.error(`[KeystrokeCapture] Element #${inputElementId} not found`);
            return;
        }

        if (this._boundKeyDown) input.removeEventListener('keydown', this._boundKeyDown);
        if (this._boundKeyUp)   input.removeEventListener('keyup',   this._boundKeyUp);

        this.reset();
        this.isCapturing = true;

        this._boundKeyDown = (e) => this.onKeyDown(e);
        this._boundKeyUp   = (e) => this.onKeyUp(e);

        input.addEventListener('keydown', this._boundKeyDown);
        input.addEventListener('keyup',   this._boundKeyUp);
        console.log('[KeystrokeCapture] Attached to:', inputElementId);
    },

    onKeyDown(e) {
        if (!this.isCapturing) return;
        const now = performance.now();

        if (this.startTime === null) this.startTime = now;

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
                if (dwell >= 0 && dwell < 2000) {
                    dwellTimes.push(dwell);
                    pairedPress.push(press);
                }
            }
        }
        return { dwellTimes, pairedPress };
    },

    // Returns true if typed text is within 30% edit distance of the target phrase.
    // Prevents grossly misspelled attempts from corrupting the model.
    _phraseConsistencyOk(inputValue) {
        const target = this.targetPhrase;
        const typed  = inputValue.trim().toLowerCase();
        if (typed.length === 0) return false;

        const m = target.length, n = typed.length;
        const dp = Array.from({ length: m + 1 }, (_, i) => [i]);
        for (let j = 1; j <= n; j++) dp[0][j] = j;
        for (let i = 1; i <= m; i++) {
            for (let j = 1; j <= n; j++) {
                dp[i][j] = target[i - 1] === typed[j - 1]
                    ? dp[i - 1][j - 1]
                    : 1 + Math.min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1]);
            }
        }
        const dist  = dp[m][n];
        const ratio = dist / Math.max(m, n);
        if (ratio > 0.30) {
            console.warn(
                '[KeystrokeCapture] Phrase edit distance ' +
                (ratio * 100).toFixed(1) + '% > 30% — rejecting sample'
            );
            return false;
        }
        return true;
    },

    extractFeatures() {
        const pressEvents   = this.events.filter(e => e.type === 'press');
        const releaseEvents = this.events.filter(e => e.type === 'release');

        if (pressEvents.length < 5) {
            console.warn('[KeystrokeCapture] Too few press events:', pressEvents.length);
            return null;
        }

        const { dwellTimes } = this._buildDwellPairs();

        // Flight times: release[i] → press[i+1]
        const flightTimes = [];
        for (let i = 0; i < releaseEvents.length - 1; i++) {
            if (i < pressEvents.length - 1) {
                const flight = pressEvents[i + 1].time - releaseEvents[i].time;
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

        // Digraphs
        const digraphMap = {};
        this.trackedDigraphs.forEach(dg => { digraphMap[dg] = []; });
        for (let i = 0; i < pressEvents.length - 1; i++) {
            const pair = (pressEvents[i].key + pressEvents[i + 1].key).toLowerCase();
            if (digraphMap[pair] !== undefined) {
                const dt = pressEvents[i + 1].time - pressEvents[i].time;
                if (dt >= 0 && dt < 3000) digraphMap[pair].push(dt);
            }
        }

        // Shift-lag: time between Shift keydown and the next character keydown
        const shiftLags = [];
        for (let i = 0; i < pressEvents.length - 1; i++) {
            if (pressEvents[i].key === 'Shift') {
                const nextChar = pressEvents[i + 1];
                if (nextChar && nextChar.key.length === 1) {
                    const lag = nextChar.time - pressEvents[i].time;
                    if (lag >= 0 && lag < 500) shiftLags.push(lag);
                }
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

        // Hand alternation
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

        const raw = {
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

            ...Object.fromEntries(
                this.trackedDigraphs.map(dg => [
                    `digraph_${dg}`, _mean(digraphMap[dg])
                ])
            ),

            typing_speed_cpm:        typingSpeedCpm,
            typing_duration:         duration,
            rhythm_mean:             rhythmMean,
            rhythm_std:              rhythmStd,
            rhythm_cv:               rhythmCv,
            pause_count:             pauses.length,
            pause_mean:              _mean(pauses),
            backspace_ratio:         backspaceRatio,
            backspace_count:         this.backspaceCount,
            hand_alternation_ratio:  text.length > 1 ? alternations / (text.length - 1) : 0,
            same_hand_sequence_mean: _mean(sameHandSeqs),
            finger_transition_ratio: text.length > 1 ? fingerTransitions / (text.length - 1) : 0,
            seek_time_mean:          _mean(seekTimes),
            seek_time_count:         seekTimes.length,

            shift_lag_mean:  _mean(shiftLags),
            shift_lag_std:   _std(shiftLags),
            shift_lag_count: shiftLags.length,

            // Raw arrays for database storage
            dwell_times:  dwellTimes,
            flight_times: flightTimes,
            typing_speed: typingSpeedCpm / 60,
        };

        const features = this._normalize(raw);
        console.log(`[KeystrokeCapture] Features extracted: ${Object.keys(features).length}`);
        return features;
    },

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
            shift_lag_norm:   f.shift_lag_mean > 0 ? f.shift_lag_mean / baseline : 0,
        };
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
    },
};

// ── Math helpers ──────────────────────────────────────────────────────────────
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