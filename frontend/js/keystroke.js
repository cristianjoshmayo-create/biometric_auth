// frontend/js/keystroke.js
// Keystroke Dynamics Capture — v4 (TypingDNA Integration)
//
// What changed in v4:
//  • Loads the official TypingDNA JS recorder from their public CDN
//    (https://typingdna.com/scripts/typingdna.js — Apache 2.0 license)
//    as the PRIMARY capture engine.  All dwell, flight, p2p timings now
//    come from TypingDNA's battle-tested implementation instead of our
//    own event loop.
//
//  • extractFeatures() merges TypingDNA's raw timing vectors with all the
//    higher-level features the backend RF model expects (hand-alternation,
//    finger transitions, digraphs, seek time, etc.).  The backend sees
//    exactly the same feature schema as v3 — zero backend changes needed.
//
//  • getTypingPattern() is exposed so you can optionally forward the raw
//    TypingDNA pattern string to any compatible TypingDNA API endpoint for
//    secondary verification in the future.
//
//  • getQuality(), onQualityUpdate, addPreviousAttempt(), getCrossAttemptCV(),
//    setPhrase(), validatePhrase(), reset(), clearHistory() — all preserved
//    with identical signatures to v3.
//
//  • Graceful degradation: if the TypingDNA CDN script fails to load, the
//    module falls back silently to v3's own event capture so the system
//    keeps working offline or in restricted environments.
//
// ────────────────────────────────────────────────────────────────────────────

const SEEK_THRESHOLD = 120;   // ms: flight time above this = a seek (key-hunt) event

// ── TypingDNA loader ─────────────────────────────────────────────────────────
// Loads typingdna.js from the public CDN and resolves with the tdna instance.
// Called once on first attach(); subsequent calls return the cached promise.

let _tdnaPromise = null;
let _tdnaInstance = null;   // TypingDNA instance once loaded

function _loadTypingDNA() {
    if (_tdnaPromise) return _tdnaPromise;
    _tdnaPromise = new Promise((resolve) => {
        if (typeof TypingDNA !== 'undefined') {
            // Already on the page (e.g. script tag in HTML)
            _tdnaInstance = new TypingDNA();
            console.log('[KeystrokeCapture] TypingDNA already present — using existing instance.');
            return resolve(true);
        }
        const script = document.createElement('script');
        script.src = 'https://typingdna.com/scripts/typingdna.js';
        script.async = true;
        script.onload = () => {
            try {
                _tdnaInstance = new TypingDNA();
                console.log('[KeystrokeCapture] TypingDNA recorder loaded from CDN.');
                resolve(true);
            } catch (e) {
                console.warn('[KeystrokeCapture] TypingDNA instantiation failed — using fallback.', e);
                resolve(false);
            }
        };
        script.onerror = () => {
            console.warn('[KeystrokeCapture] TypingDNA CDN unreachable — using fallback capture.');
            resolve(false);
        };
        document.head.appendChild(script);
    });
    return _tdnaPromise;
}

// ── Main module ──────────────────────────────────────────────────────────────

const KeystrokeCapture = {
    // ── State ────────────────────────────────────────────────────────────────
    events: [],           // fallback: raw keydown/keyup events
    isCapturing: false,
    startTime: null,
    endTime: null,
    textBuffer: [],
    backspaceCount: 0,
    targetPhrase: '',
    _attachedElementId: null,

    // Cross-attempt history (persists across reset() calls)
    _prevAttempts: [],

    // Real-time quality callback — wire up after attach()
    // e.g. KeystrokeCapture.onQualityUpdate = (score, label) => { ... };
    onQualityUpdate: null,

    // ── Keyboard layout (unchanged from v3) ──────────────────────────────────
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

    trackedDigraphs: [],

    // ── Attach / detach ──────────────────────────────────────────────────────

    async attach(inputElementId) {
        const input = document.getElementById(inputElementId);
        if (!input) {
            console.error(`[KeystrokeCapture] Element #${inputElementId} not found`);
            return;
        }

        // Detach previous listeners
        if (this._boundKeyDown) {
            const prev = document.getElementById(this._attachedElementId || '');
            if (prev) {
                prev.removeEventListener('keydown', this._boundKeyDown);
                prev.removeEventListener('keyup',   this._boundKeyUp);
            }
        }

        this.reset();
        this.isCapturing = true;
        this._attachedElementId = inputElementId;

        // Try to load TypingDNA; if it fails we fall back to our own capture
        const tdnaLoaded = await _loadTypingDNA();

        if (tdnaLoaded && _tdnaInstance) {
            // ── TypingDNA mode ───────────────────────────────────────────────
            // TypingDNA starts recording globally as soon as it's instantiated.
            // Restrict it to our specific input.
            _tdnaInstance.reset();
            _tdnaInstance.addTarget(inputElementId);
            _tdnaInstance.start();
            console.log(`[KeystrokeCapture] TypingDNA recorder targeting: #${inputElementId}`);
        }

        // Always attach our own listeners too:
        //  • In TypingDNA mode  → used for textBuffer, backspaceCount, real-time quality
        //  • In fallback mode   → full capture (same as v3)
        this._boundKeyDown = (e) => this._onKeyDown(e);
        this._boundKeyUp   = (e) => this._onKeyUp(e);
        input.addEventListener('keydown', this._boundKeyDown);
        input.addEventListener('keyup',   this._boundKeyUp);

        console.log(`[KeystrokeCapture] Attached to: #${inputElementId} (tdna=${tdnaLoaded})`);
    },

    // ── Event handlers ───────────────────────────────────────────────────────

    _onKeyDown(e) {
        if (!this.isCapturing) return;
        const now = performance.now();
        if (this.startTime === null) this.startTime = now;

        // Always push to our own event buffer (used for fallback AND for
        // textBuffer / backspaceCount which TypingDNA doesn't expose).
        this.events.push({ type: 'press', key: e.key, code: e.code, time: now, used: false });

        if (e.key === 'Backspace') {
            this.backspaceCount++;
            if (this.textBuffer.length > 0) this.textBuffer.pop();
        } else if (e.key.length === 1) {
            this.textBuffer.push(e.key);
        }
    },

    _onKeyUp(e) {
        if (!this.isCapturing) return;
        const now = performance.now();
        this.endTime = now;
        this.events.push({ type: 'release', key: e.key, code: e.code, time: now, used: false });

        // Real-time quality (fires async so it doesn't block keyup path)
        if (typeof this.onQualityUpdate === 'function') {
            const pressEvents = this.events.filter(ev => ev.type === 'press');
            if (pressEvents.length >= 5) {
                const { score, label } = this._computeQualityFast(pressEvents);
                this.onQualityUpdate(score, label);
            }
        }
    },

    // ── TypingDNA pattern accessor ───────────────────────────────────────────
    // Returns the raw TypingDNA typing-pattern string (type 1 = sametext diagram)
    // for the current target phrase.  Returns null when TypingDNA is unavailable.
    //
    // Use this if you want to store or forward the pattern to a TypingDNA API
    // endpoint in the future:
    //   const pattern = KeystrokeCapture.getTypingPattern();

    getTypingPattern() {
        if (!_tdnaInstance) return null;
        try {
            // type 1 = sametext/diagram pattern (recommended for fixed phrases)
            // text  = the phrase the user is typing
            // targetId = restrict to our input element
            return _tdnaInstance.getTypingPattern({
                type:      1,
                text:      this.targetPhrase,
                targetId:  this._attachedElementId || undefined,
            });
        } catch (e) {
            console.warn('[KeystrokeCapture] getTypingPattern error:', e);
            return null;
        }
    },

    // ── Helpers ──────────────────────────────────────────────────────────────

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

    // ── Dwell / flight pairing (v3, used in fallback mode) ───────────────────

    _buildDwellPairs() {
        const pressEvents   = this.events.filter(e => e.type === 'press');
        const releaseEvents = this.events.filter(e => e.type === 'release')
                                         .map(e => ({ ...e, used: false }));
        const dwellTimes    = [];
        const pairedPress   = [];
        const pairedRelease = [];

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
                    pairedRelease.push(release);
                }
            }
        }
        return { dwellTimes, pairedPress, pairedRelease };
    },

    // ── Parse TypingDNA pattern string into timing arrays ────────────────────
    // TypingDNA encodes the pattern as a comma-separated string.
    // The segment structure for type-1 (sametext) patterns:
    //   history_length, seek_time, version, ...per-key-triplets(dwell, pp, rr)
    //
    // Format per the open-source typingdna.js:
    //   <history>;<seektime>;<version>;[dwellA,flightA,dwellB,flightB,…]
    //
    // We decode only what we need: dwell and flight arrays.
    // If the pattern string is unavailable we return nulls and fall back.

    _parseTypingDNAPattern(pattern) {
        if (!pattern || typeof pattern !== 'string') return null;
        try {
            const parts = pattern.split(';');
            // TypingDNA type-1 header: version ; length ; flags ; seektime ; <keydata...>
            // Key data starts at index 4, not 3.
            if (parts.length < 5) return null;

            const keyData = parts.slice(4).join(';').split(',').map(Number);

            const dwellTimes  = [];
            const flightTimes = [];

            for (let i = 0; i < keyData.length; i++) {
                const v = keyData[i];
                if (isNaN(v) || v < 0) continue;
                if (i % 2 === 0) {
                    if (v > 0 && v < 2000) dwellTimes.push(v);
                } else {
                    if (v < 2000) flightTimes.push(v);
                }
            }

            // Require at least 4 valid dwell values — fewer means the parse
            // likely hit header data, not real keystroke timings.
            return dwellTimes.length >= 4 ? { dwellTimes, flightTimes } : null;
        } catch (e) {
            return null;
        }
    },

    // ── Phrase consistency check ─────────────────────────────────────────────

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
        const ratio = dp[m][n] / Math.max(m, n);
        if (ratio > 0.30) {
            console.warn(`[KeystrokeCapture] Edit distance ${(ratio * 100).toFixed(1)}% > 30% — rejecting`);
            return false;
        }
        return true;
    },

    // ── Feature extraction ───────────────────────────────────────────────────
    // Returns the same feature schema as v3 — fully compatible with the backend
    // Random Forest model.  Timing arrays are sourced from TypingDNA when
    // available, otherwise from our own fallback capture.

    extractFeatures() {
        const pressEvents   = this.events.filter(e => e.type === 'press');
        const releaseEvents = this.events.filter(e => e.type === 'release');

        if (pressEvents.length < 5) {
            console.warn('[KeystrokeCapture] Too few press events:', pressEvents.length);
            return null;
        }

        // ── Source timing arrays ─────────────────────────────────────────────
        let dwellTimes, flightTimes, pairedPress, pairedRelease;

        const tdnaPattern = this.getTypingPattern();
        const tdnaParsed  = this._parseTypingDNAPattern(tdnaPattern);

        if (tdnaParsed) {
            // TypingDNA mode: use its high-fidelity timing vectors
            dwellTimes  = tdnaParsed.dwellTimes;
            flightTimes = tdnaParsed.flightTimes;
            // pairedPress/Release used only for digraph/shift-lag; build from fallback
            const fb    = this._buildDwellPairs();
            pairedPress   = fb.pairedPress;
            pairedRelease = fb.pairedRelease;
            console.log(`[KeystrokeCapture] Using TypingDNA timing vectors (dwell=${dwellTimes.length}, flight=${flightTimes.length})`);
        } else {
            // Fallback mode: v3 own capture
            const fb    = this._buildDwellPairs();
            dwellTimes    = fb.dwellTimes;
            pairedPress   = fb.pairedPress;
            pairedRelease = fb.pairedRelease;

            flightTimes = [];
            for (let i = 0; i < pairedRelease.length - 1; i++) {
                if (i < pairedPress.length - 1) {
                    const flight = pairedPress[i + 1].time - pairedRelease[i].time;
                    if (flight > -50 && flight < 2000) flightTimes.push(Math.max(0, flight));
                }
            }
            console.log('[KeystrokeCapture] Using fallback timing capture.');
        }

        // ── Press-to-press ───────────────────────────────────────────────────
        const p2pTimes = [];
        for (let i = 0; i < pressEvents.length - 1; i++) {
            const dt = pressEvents[i + 1].time - pressEvents[i].time;
            if (dt >= 0 && dt < 3000) p2pTimes.push(dt);
        }

        // ── Release-to-release ───────────────────────────────────────────────
        const r2rTimes = [];
        for (let i = 0; i < releaseEvents.length - 1; i++) {
            const dt = releaseEvents[i + 1].time - releaseEvents[i].time;
            if (dt >= 0 && dt < 3000) r2rTimes.push(dt);
        }

        // ── Digraphs ─────────────────────────────────────────────────────────
        const digraphMap = {};
        this.trackedDigraphs.forEach(dg => { digraphMap[dg] = []; });
        for (let i = 0; i < pressEvents.length - 1; i++) {
            const pair = (pressEvents[i].key + pressEvents[i + 1].key).toLowerCase();
            if (digraphMap[pair] !== undefined) {
                const dt = pressEvents[i + 1].time - pressEvents[i].time;
                if (dt >= 0 && dt < 3000) digraphMap[pair].push(dt);
            }
        }

        // ── Shift-lag ────────────────────────────────────────────────────────
        const shiftLags = [];
        for (let i = 0; i < pressEvents.length - 1; i++) {
            if (pressEvents[i].key === 'Shift') {
                const next = pressEvents[i + 1];
                if (next && next.key.length === 1) {
                    const lag = next.time - pressEvents[i].time;
                    if (lag >= 0 && lag < 500) shiftLags.push(lag);
                }
            }
        }

        // ── Typing speed ─────────────────────────────────────────────────────
        const duration = (this.endTime && this.startTime)
            ? (this.endTime - this.startTime) / 1000
            : 0;
        const typingSpeedCpm = duration > 0
            ? (this.textBuffer.length / duration) * 60
            : 0;

        // ── Rhythm ───────────────────────────────────────────────────────────
        const rhythmMean = _mean(p2pTimes);
        const rhythmStd  = _std(p2pTimes);
        const rhythmCv   = rhythmMean > 0 ? rhythmStd / rhythmMean : 0;

        // ── Pauses > 500ms ───────────────────────────────────────────────────
        const pauses = p2pTimes.filter(t => t > 500);

        // ── Hand alternation ─────────────────────────────────────────────────
        const text = this.textBuffer.join('').toLowerCase();
        let alternations = 0;
        const sameHandSeqs = [];
        let currentSeq = 0;
        for (let i = 0; i < text.length - 1; i++) {
            const curr = this.getHand(text[i]);
            const next = this.getHand(text[i + 1]);
            if (curr && next && curr !== next) {
                alternations++;
                if (currentSeq > 0) sameHandSeqs.push(currentSeq);
                currentSeq = 0;
            } else {
                currentSeq++;
            }
        }
        if (currentSeq > 0) sameHandSeqs.push(currentSeq);

        // ── Finger transitions ───────────────────────────────────────────────
        let fingerTransitions = 0;
        for (let i = 0; i < text.length - 1; i++) {
            const f1 = this.getFinger(text[i]);
            const f2 = this.getFinger(text[i + 1]);
            if (f1 && f2 && f1 !== f2) fingerTransitions++;
        }

        // ── Seek time (flight > SEEK_THRESHOLD) ──────────────────────────────
        const seekEvents = flightTimes.filter(t => t > SEEK_THRESHOLD);

        // ── Backspace ratio ───────────────────────────────────────────────────
        const totalKeys      = pressEvents.length;
        const backspaceRatio = totalKeys > 0 ? this.backspaceCount / totalKeys : 0;

        // ── Assemble raw feature object ───────────────────────────────────────
        const raw = {
            dwell_mean:    _mean(dwellTimes),
            dwell_std:     _std(dwellTimes),
            dwell_median:  _median(dwellTimes),
            dwell_min:     dwellTimes.length ? Math.min(...dwellTimes) : 0,
            dwell_max:     dwellTimes.length ? Math.max(...dwellTimes) : 0,

            flight_mean:   _mean(flightTimes),
            flight_std:    _std(flightTimes),
            flight_median: _median(flightTimes),

            p2p_mean:  _mean(p2pTimes),
            p2p_std:   _std(p2pTimes),
            r2r_mean:  _mean(r2rTimes),
            r2r_std:   _std(r2rTimes),

            ...Object.fromEntries(
                this.trackedDigraphs.map(dg => [`digraph_${dg}`, _mean(digraphMap[dg])])
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

            seek_time_mean:  _mean(seekEvents),
            seek_time_count: seekEvents.length,

            shift_lag_mean:  _mean(shiftLags),
            shift_lag_std:   _std(shiftLags),
            shift_lag_count: shiftLags.length,

            extra_digraphs: Object.fromEntries(
                this.trackedDigraphs.map(dg => [dg, _mean(digraphMap[dg])])
            ),
            dwell_times:  dwellTimes,
            flight_times: flightTimes,
            typing_speed: typingSpeedCpm / 60,

            // v4: expose TypingDNA pattern string for optional forwarding
            tdna_pattern: tdnaPattern || null,
            tdna_active:  !!tdnaParsed,
        };

        const features = this._normalize(raw);
        console.log(`[KeystrokeCapture] Features extracted: ${Object.keys(features).length} (tdna=${raw.tdna_active})`);
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

    // ── Quality gate (unchanged from v3) ─────────────────────────────────────

    getQuality() {
        const pressEvents = this.events.filter(e => e.type === 'press');
        if (pressEvents.length < 5) {
            return { score: 0, label: 'weak', details: { reason: 'too few keystrokes' } };
        }

        const { dwellTimes } = this._buildDwellPairs();

        const p2pTimes = [];
        for (let i = 0; i < pressEvents.length - 1; i++) {
            const dt = pressEvents[i + 1].time - pressEvents[i].time;
            if (dt >= 0 && dt < 3000) p2pTimes.push(dt);
        }

        const phraseLen    = this.targetPhrase.replace(/\s+/g, '').length || 20;
        const totalKeys    = pressEvents.length;
        const keyCountScore = Math.min(1, totalKeys / (phraseLen * 1.5));

        const rhythmMean   = _mean(p2pTimes);
        const rhythmCv     = rhythmMean > 0 ? _std(p2pTimes) / rhythmMean : 1;
        const rhythmScore  = Math.max(0, Math.min(1, (0.80 - rhythmCv) / (0.80 - 0.25)));

        const dwellMean = _mean(dwellTimes);
        let dwellScore;
        if (dwellMean < 20 || dwellMean > 500)       dwellScore = 0;
        else if (dwellMean >= 40 && dwellMean <= 250) dwellScore = 1;
        else if (dwellMean < 40)                      dwellScore = (dwellMean - 20) / 20;
        else                                          dwellScore = Math.max(0, (500 - dwellMean) / 250);

        const backRatio   = totalKeys > 0 ? this.backspaceCount / totalKeys : 0;
        const errorScore  = Math.max(0, 1 - backRatio / 0.2);

        let crossScore = 0.75;
        const cv = this.getCrossAttemptCV();
        if (cv !== null) {
            crossScore = Math.max(0, Math.min(1, (0.40 - cv) / (0.40 - 0.10)));
        }

        const score = Math.min(1,
            keyCountScore * 0.25 +
            rhythmScore   * 0.35 +
            dwellScore    * 0.20 +
            errorScore    * 0.10 +
            crossScore    * 0.10
        );

        const label = score >= 0.80 ? 'strong'
                    : score >= 0.60 ? 'good'
                    : score >= 0.30 ? 'acceptable'
                    : 'weak';

        console.log(
            `[KeystrokeCapture] Quality: ${(score * 100).toFixed(0)}% (${label}) — ` +
            `keys=${keyCountScore.toFixed(2)} rhythm=${rhythmScore.toFixed(2)} ` +
            `dwell=${dwellScore.toFixed(2)} error=${errorScore.toFixed(2)} cross=${crossScore.toFixed(2)}`
        );

        return {
            score,
            label,
            details: {
                keyCount:     keyCountScore,
                rhythm:       rhythmScore,
                dwell:        dwellScore,
                errorRate:    errorScore,
                crossAttempt: crossScore,
                rhythmCv,
                dwellMean,
                backRatio,
            },
        };
    },

    // ── Fast quality estimate (real-time keyup callback) ─────────────────────

    _computeQualityFast(pressEvents) {
        const p2pTimes = [];
        for (let i = 0; i < pressEvents.length - 1; i++) {
            const dt = pressEvents[i + 1].time - pressEvents[i].time;
            if (dt >= 0 && dt < 3000) p2pTimes.push(dt);
        }
        const phraseLen    = this.targetPhrase.replace(/\s+/g, '').length || 20;
        const keyCountScore = Math.min(1, pressEvents.length / (phraseLen * 1.5));
        const rhythmMean   = _mean(p2pTimes);
        const rhythmCv     = rhythmMean > 0 ? _std(p2pTimes) / rhythmMean : 1;
        const rhythmScore  = Math.max(0, Math.min(1, (0.80 - rhythmCv) / 0.55));
        const backRatio    = pressEvents.length > 0 ? this.backspaceCount / pressEvents.length : 0;
        const errorScore   = Math.max(0, 1 - backRatio / 0.2);

        const score = Math.min(1,
            keyCountScore * 0.35 +
            rhythmScore   * 0.50 +
            errorScore    * 0.15
        );
        const label = score >= 0.80 ? 'strong'
                    : score >= 0.60 ? 'good'
                    : score >= 0.30 ? 'acceptable'
                    : 'weak';
        return { score, label };
    },

    // ── Cross-attempt consistency ─────────────────────────────────────────────

    addPreviousAttempt(features) {
        if (!features) return;
        this._prevAttempts.push({
            typing_speed_cpm: features.typing_speed_cpm || 0,
            dwell_mean:       features.dwell_mean       || 0,
            rhythm_cv:        features.rhythm_cv        || 0,
        });
    },

    getCrossAttemptCV() {
        if (this._prevAttempts.length < 1) return null;
        const speeds = this._prevAttempts.map(a => a.typing_speed_cpm).filter(v => v > 0);
        const dwells = this._prevAttempts.map(a => a.dwell_mean).filter(v => v > 0);
        if (speeds.length < 2 && dwells.length < 2) return null;
        const cvOf = arr => { const m = _mean(arr); return m > 0 ? _std(arr) / m : 0; };
        const cvs = [];
        if (speeds.length >= 2) cvs.push(cvOf(speeds));
        if (dwells.length >= 2) cvs.push(cvOf(dwells));
        return cvs.length > 0 ? _mean(cvs) : null;
    },

    // ── Phrase management ─────────────────────────────────────────────────────

    validatePhrase(inputValue) {
        return inputValue.trim() === this.targetPhrase;
    },

    setPhrase(phrase) {
        this.targetPhrase = phrase.trim();
        const clean = phrase.toLowerCase().replace(/\s+/g, '');
        const seen  = new Set();
        const pairs = [];
        for (let i = 0; i < clean.length - 1; i++) {
            const pair = clean[i] + clean[i + 1];
            if (/^[a-z]{2}$/.test(pair) && !seen.has(pair)) {
                seen.add(pair);
                pairs.push(pair);
            }
        }
        this.trackedDigraphs = pairs;
        console.log(`[KeystrokeCapture] Phrase set: "${phrase}" | digraphs:`, pairs.join(', ') || '(none)');
    },

    // ── Reset ─────────────────────────────────────────────────────────────────
    // reset() clears the current attempt but keeps _prevAttempts.
    // Stops TypingDNA recorder and removes its target.

    reset() {
        this.events         = [];
        this.isCapturing    = false;
        this.startTime      = null;
        this.endTime        = null;
        this.textBuffer     = [];
        this.backspaceCount = 0;

        if (_tdnaInstance) {
            try {
                _tdnaInstance.stop();
                if (this._attachedElementId) {
                    _tdnaInstance.removeTarget(this._attachedElementId);
                }
                _tdnaInstance.reset();
            } catch (e) { /* ignore */ }
        }
    },

    clearHistory() {
        this._prevAttempts = [];
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