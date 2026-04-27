// frontend/js/keystroke.js
// Keystroke Dynamics Capture — v3
//
// New in v3 (TypingDNA-inspired improvements):
//  1. getQuality()            — 0–1 quality score computed from the captured
//                               event stream.  Mirrors TypingDNA's pattern
//                               quality signal.  Call after typing is done,
//                               before submitting.  Scores:
//                                 < 0.30  → weak   (reject / prompt retry)
//                                 0.30–0.59 → acceptable
//                                 0.60–0.79 → good
//                                 ≥ 0.80  → strong
//  2. onQualityUpdate(score, label) — real-time callback fired on every keyup.
//                               Wire it to a progress bar so the user sees
//                               quality grow as they type, not just after
//                               submission.  Default is a no-op.
//  3. Seek-time fix           — "seek time" is the pause between releasing one
//                               key and pressing the next (i.e. flight time).
//                               The old code counted p2p > 300ms, which is
//                               actually press-to-press and mixes seek with
//                               the previous key's still-held dwell.  v3 uses
//                               flight_time > SEEK_THRESHOLD (120ms) — cleaner
//                               measure of hesitation / key-hunt events.
//  4. addPreviousAttempt() /  — cross-attempt consistency tracking.  After
//     getCrossAttemptCV()       each successful enrollment attempt call
//                               addPreviousAttempt(features).  Before the next
//                               attempt, getCrossAttemptCV() returns the CV
//                               of typing_speed_cpm and dwell_mean across all
//                               previous attempts — a direct consistency metric
//                               the backend already stores but that you can now
//                               surface on the frontend as a "consistency" score.
//  5. _buildDwellPairs()      — unchanged but now also returns releaseEvents
//                               so extractFeatures() can compute seek time from
//                               flight windows without re-scanning.

const SEEK_THRESHOLD = 120;   // ms: flight time above this = a seek (key-hunt) event

const KeystrokeCapture = {
    events: [],
    isCapturing: false,
    startTime: null,
    endTime: null,
    textBuffer: [],
    backspaceCount: 0,
    targetPhrase: "",

    // ── Cross-attempt history ────────────────────────────────────────────────
    // Populated by addPreviousAttempt() after each successful submission.
    // Used by getCrossAttemptCV() and quality scoring.
    _prevAttempts: [],     // [{typing_speed_cpm, dwell_mean, rhythm_cv}, …]

    // ── Real-time quality callback ───────────────────────────────────────────
    // Override this after attach() to wire up a progress bar / status label.
    // Called on every keyup.  score is 0–1, label is one of the four strings
    // below.  Both args are null until at least 5 press events are recorded.
    //
    // Example:
    //   KeystrokeCapture.onQualityUpdate = (score, label) => {
    //       bar.style.width = (score * 100) + '%';
    //       badge.textContent = label;
    //   };
    onQualityUpdate: null,

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
    trackedTrigraphs: [],

    // ── Attach / detach ──────────────────────────────────────────────────────
    //
    // Capture lifecycle:
    //   attach()  → registers listeners but does NOT start capturing yet.
    //   focus     → clears any stale data, sets isCapturing = true, shows indicator.
    //   blur      → pauses capturing (isCapturing = false).  Data is kept so the
    //               user can re-focus and continue typing the phrase.
    //   reset()   → called between attempts or on a fresh start; clears everything.
    //
    // This prevents two common bugs:
    //   1. Auto-focused input silently captures keystrokes before the user is ready.
    //   2. Accidental keystrokes typed while the user thinks they are elsewhere
    //      contaminating the timing data.

    attach(inputElementId) {
        const input = document.getElementById(inputElementId);
        if (!input) {
            console.error(`[KeystrokeCapture] Element #${inputElementId} not found`);
            return;
        }

        // Remove any previous listeners on this element
        if (this._boundKeyDown) input.removeEventListener('keydown', this._boundKeyDown);
        if (this._boundKeyUp)   input.removeEventListener('keyup',   this._boundKeyUp);
        if (this._boundFocus)   input.removeEventListener('focus',   this._boundFocus);
        if (this._boundBlur)    input.removeEventListener('blur',    this._boundBlur);

        this.reset();
        this.isCapturing = false;   // NOT capturing yet — wait for intentional focus

        this._boundKeyDown = (e) => this.onKeyDown(e);
        this._boundKeyUp   = (e) => this.onKeyUp(e);

        this._boundFocus = () => {
            // User intentionally clicked / tabbed into the field.
            // Always clear and restart — timing data must be one continuous
            // session.  If the user clicked away mid-phrase, any events
            // captured before that point have a meaningless time gap in them.
            const hadData = this.events.length > 0;
            this.events         = [];
            this.startTime      = null;
            this.endTime        = null;
            this.textBuffer     = [];
            this.backspaceCount = 0;
            this.isCapturing    = true;
            this._updateFocusIndicator(input, true, hadData);
            // Also clear the input value so the user retypes cleanly
            if (hadData) input.value = '';
            console.log('[KeystrokeCapture] Focus — capture started fresh' + (hadData ? ' (previous data cleared)' : ''));
        };

        this._boundBlur = () => {
            // User clicked away — pause capture so stray keystrokes don't leak in.
            this.isCapturing = false;
            this._updateFocusIndicator(input, false);
            console.log('[KeystrokeCapture] Blur — capture paused');
        };

        input.addEventListener('keydown', this._boundKeyDown);
        input.addEventListener('keyup',   this._boundKeyUp);
        input.addEventListener('focus',   this._boundFocus);
        input.addEventListener('blur',    this._boundBlur);

        this._attachedInputId = inputElementId;

        // Auto-focus so the user can start typing without clicking first.
        // Safe because startTime is only set on the first real keydown —
        // pre-typing idle time is never recorded. The focus event handler
        // above resets state and sets isCapturing=true, same as a manual click.
        // If the element is already the active element (e.g. caller focused it
        // before calling attach()), input.focus() is a no-op and the focus
        // event never fires — invoke the handler directly so isCapturing flips.
        if (document.activeElement === input) {
            this._boundFocus();
        } else {
            input.focus();
        }

        console.log('[KeystrokeCapture] Attached to:', inputElementId, '— focused, waiting for input');
    },

    // Show / hide a small visual indicator so the user always knows when capture
    // is active.  Looks for a sibling element with class "ks-capture-indicator";
    // if none exists, adds a small status line below the input automatically.
    _updateFocusIndicator(input, active, wasReset = false) {
        let indicator = input.parentElement
            ? input.parentElement.querySelector('.ks-capture-indicator')
            : null;

        if (!indicator) {
            indicator = document.createElement('p');
            indicator.className = 'ks-capture-indicator text-xs mt-1 text-center';
            if (input.parentElement) input.parentElement.appendChild(indicator);
        }

        if (active && wasReset) {
            indicator.textContent = '🔴 Restarted — retype your phrase from the beginning';
            indicator.style.color = '#f59e0b';   // amber — signals a restart
        } else if (active) {
            indicator.textContent = '🔴 Capturing — type your phrase now';
            indicator.style.color = '#10b981';   // green
        } else {
            indicator.textContent = '⏸ Click the box above to start typing';
            indicator.style.color = '#6b7280';   // gray
        }
    },

    // ── Event handlers ───────────────────────────────────────────────────────

    onKeyDown(e) {
        if (!this.isCapturing) return;
        // Reject synthetic events (dispatchEvent, DevTools, automation tools).
        // Only real user keystrokes have isTrusted === true.
        if (!e.isTrusted) return;
        // Ignore OS-level auto-repeat: a held key fires repeated keydowns that
        // corrupt dwell and flood p2p/flight with ~0ms intervals.
        if (e.repeat) return;
        const now = performance.now();
        if (this.startTime === null) this.startTime = now;

        this.events.push({ type: 'press', key: e.key, code: e.code, time: now, used: false });

        if (e.key === 'Backspace') {
            this.backspaceCount++;
            if (this.textBuffer.length > 0) this.textBuffer.pop();
        } else if (e.key.length === 1) {
            this.textBuffer.push(e.key);
        }
    },

    onKeyUp(e) {
        if (!this.isCapturing) return;
        if (!e.isTrusted) return;
        const now = performance.now();
        this.endTime = now;
        this.events.push({ type: 'release', key: e.key, code: e.code, time: now, used: false });

        // Real-time quality update — fire asynchronously so it doesn't block
        // the keyup path.  Only fires when the callback is wired up.
        if (typeof this.onQualityUpdate === 'function') {
            const pressEvents = this.events.filter(ev => ev.type === 'press');
            if (pressEvents.length >= 5) {
                const { score, label } = this._computeQualityFast(pressEvents);
                this.onQualityUpdate(score, label);
            }
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

    // ── Dwell / flight pairing ───────────────────────────────────────────────
    // Returns { dwellTimes, pairedPress, pairedRelease } so extractFeatures
    // can compute seek/flight from the same release events without re-scanning.

    // ── Same-text pattern: clean press sequence ─────────────────────────────
    // Replays the event stream like a stack. Each normal character pushes its
    // press event; each Backspace pops the last one. The result is only the
    // press events whose characters survived.
    //
    // Each surviving event gets an `adjacentToPrev` flag:
    //   true  → no backspace occurred between this character and the previous
    //            surviving one — the p2p/digraph interval is trustworthy.
    //   false → a backspace happened in between — the timing gap is inflated
    //            by the correction detour and must NOT be used for p2p/digraphs.
    //
    // Example: typing "txhe" (x → backspace → h):
    //   clean sequence = [t, h, e]
    //   t.adjacentToPrev = false  (first character)
    //   h.adjacentToPrev = false  (backspace occurred between t and h)
    //   e.adjacentToPrev = true   (h → e was clean)
    //   → only the h→e digraph/p2p interval is used; t→h is skipped.
    _getCleanPressEvents() {
        const allPressEvents = this.events.filter(e => e.type === 'press');
        const stack = [];
        let hadBackspaceSinceLastChar = false;

        for (const ev of allPressEvents) {
            if (ev.key === 'Backspace') {
                if (stack.length > 0) {
                    stack.pop();
                    hadBackspaceSinceLastChar = true;
                }
            } else if (ev.key.length === 1) {
                stack.push({ ...ev, adjacentToPrev: stack.length > 0 && !hadBackspaceSinceLastChar });
                hadBackspaceSinceLastChar = false;
            }
            // Shift and other modifiers skipped — they don't produce characters
        }
        return stack;
    },

    _buildDwellPairs(cleanPresses = null) {
        const pressEvents   = cleanPresses || this.events.filter(e => e.type === 'press');
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
            console.warn(`[KeystrokeCapture] Phrase edit distance ${(ratio * 100).toFixed(1)}% > 30% — rejecting`);
            return false;
        }
        return true;
    },

    // ── Feature extraction ───────────────────────────────────────────────────

    extractFeatures() {
        // All press events — used only for backspace ratio (needs full count)
        const allPressEvents = this.events.filter(e => e.type === 'press');

        if (allPressEvents.length < 5) {
            console.warn('[KeystrokeCapture] Too few press events:', allPressEvents.length);
            return null;
        }

        // ── Same-text pattern ────────────────────────────────────────────────
        // Timing features are computed only from press events whose characters
        // survived backspacing — identical to TypingDNA's same-text pattern.
        const cleanPressEvents = this._getCleanPressEvents();

        if (cleanPressEvents.length < 4) {
            console.warn('[KeystrokeCapture] Too few clean press events after backspace removal:', cleanPressEvents.length);
            return null;
        }

        const { dwellTimes, pairedPress, pairedRelease } = this._buildDwellPairs(cleanPressEvents);

        // Flight times: release[i] → press[i+1], adjacent pairs only
        // Skip if a backspace occurred between the two characters.
        const flightTimes = [];
        for (let i = 0; i < pairedRelease.length - 1; i++) {
            if (i + 1 >= pairedPress.length) break;
            if (!cleanPressEvents[i + 1] || !cleanPressEvents[i + 1].adjacentToPrev) continue;
            const flight = pairedPress[i + 1].time - pairedRelease[i].time;
            if (flight > -50 && flight < 2000) {
                flightTimes.push(Math.max(0, flight));
            }
        }

        // Press-to-press — only adjacent pairs (no backspace between them)
        // Non-adjacent intervals are inflated by the correction detour and skipped.
        const p2pTimes = [];
        for (let i = 0; i < cleanPressEvents.length - 1; i++) {
            if (!cleanPressEvents[i + 1].adjacentToPrev) continue;
            const dt = cleanPressEvents[i + 1].time - cleanPressEvents[i].time;
            if (dt >= 0 && dt < 3000) p2pTimes.push(dt);
        }

        // Release-to-release — only between adjacent clean pairs
        const r2rTimes = [];
        for (let i = 0; i < pairedRelease.length - 1; i++) {
            if (!cleanPressEvents[i + 1] || !cleanPressEvents[i + 1].adjacentToPrev) continue;
            const dt = pairedRelease[i + 1].time - pairedRelease[i].time;
            if (dt >= 0 && dt < 3000) r2rTimes.push(dt);
        }

        // Digraphs — only adjacent pairs (no backspace between the two characters)
        // Tracks all four timing variants (Killourhy–Maxion convention):
        //   DD = press[i]   → press[i+1]    (already used as `digraph_${dg}`)
        //   DU = press[i]   → release[i]    (per-pair dwell of first key)
        //   UD = release[i] → press[i+1]    (flight time for this specific pair)
        //   UU = release[i] → release[i+1]
        const digraphMap   = {};   // DD — kept for backward compatibility
        const digraphDD    = {};
        const digraphDU    = {};
        const digraphUD    = {};   // == per-digraph flight
        const digraphUU    = {};
        this.trackedDigraphs.forEach(dg => {
            digraphMap[dg] = [];
            digraphDD[dg]  = [];
            digraphDU[dg]  = [];
            digraphUD[dg]  = [];
            digraphUU[dg]  = [];
        });
        for (let i = 0; i < cleanPressEvents.length - 1; i++) {
            if (!cleanPressEvents[i + 1].adjacentToPrev) continue;
            const pair = (cleanPressEvents[i].key + cleanPressEvents[i + 1].key).toLowerCase();
            const p1 = cleanPressEvents[i], p2 = cleanPressEvents[i + 1];
            const dd = p2.time - p1.time;
            if (dd >= 0 && dd < 3000 && digraphMap[pair] !== undefined) {
                digraphMap[pair].push(dd);
                digraphDD[pair].push(dd);
            }
            // Pair-aligned releases via i / i+1 in pairedRelease (built from cleanPressEvents)
            const r1 = pairedRelease[i], r2 = pairedRelease[i + 1];
            if (r1 && r2 && digraphDU[pair] !== undefined) {
                const du = r1.time - p1.time;
                const ud = p2.time - r1.time;
                const uu = r2.time - r1.time;
                if (du >= 0 && du < 2000) digraphDU[pair].push(du);
                if (ud > -50 && ud < 2000) digraphUD[pair].push(Math.max(0, ud));
                if (uu >= 0 && uu < 3000) digraphUU[pair].push(uu);
            }
        }

        // Trigraphs — three adjacent clean presses, no backspace between any of them.
        // Press[i] → press[i+2] timing; captures sequence muscle memory better than digraphs.
        const trigraphMap = {};
        this.trackedTrigraphs.forEach(tg => { trigraphMap[tg] = []; });
        for (let i = 0; i < cleanPressEvents.length - 2; i++) {
            if (!cleanPressEvents[i + 1].adjacentToPrev) continue;
            if (!cleanPressEvents[i + 2].adjacentToPrev) continue;
            const tri = (cleanPressEvents[i].key +
                         cleanPressEvents[i + 1].key +
                         cleanPressEvents[i + 2].key).toLowerCase();
            if (trigraphMap[tri] !== undefined) {
                const dt = cleanPressEvents[i + 2].time - cleanPressEvents[i].time;
                if (dt >= 0 && dt < 5000) trigraphMap[tri].push(dt);
            }
        }

        // Shift-lag (from clean sequence — only shifts for surviving characters)
        const shiftLags = [];
        const allPressFull = this.events.filter(e => e.type === 'press');
        for (let i = 0; i < allPressFull.length - 1; i++) {
            if (allPressFull[i].key === 'Shift') {
                const next = allPressFull[i + 1];
                if (next && next.key.length === 1) {
                    const lag = next.time - allPressFull[i].time;
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

        // Rhythm (from clean p2p — same-text pattern)
        const rhythmMean = _mean(p2pTimes);
        const rhythmStd  = _std(p2pTimes);
        const rhythmCv   = rhythmMean > 0 ? rhythmStd / rhythmMean : 0;

        // Pauses > 500ms (clean p2p only)
        const pauses = p2pTimes.filter(t => t > 500);

        // Hand alternation (textBuffer is already the clean final text)
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

        // Finger transitions
        let fingerTransitions = 0;
        for (let i = 0; i < text.length - 1; i++) {
            const f1 = this.getFinger(text[i]);
            const f2 = this.getFinger(text[i + 1]);
            if (f1 && f2 && f1 !== f2) fingerTransitions++;
        }

        // Per-key dwell map: { 's': avgDwell, 'p': avgDwell, ... }
        // Tracks how long the user holds each individual key — more person-specific
        // than the aggregate dwell_mean because different fingers have different habits.
        const keyDwellAccum = {};
        for (let i = 0; i < pairedPress.length; i++) {
            const k = pairedPress[i].key.toLowerCase();
            if (k.length !== 1 || !/[a-z]/.test(k)) continue;
            if (!keyDwellAccum[k]) keyDwellAccum[k] = [];
            keyDwellAccum[k].push(dwellTimes[i]);
        }
        const key_dwell_map = {};
        for (const [k, times] of Object.entries(keyDwellAccum)) {
            if (times.length > 0) key_dwell_map[k] = _mean(times);
        }

        // Seek time from clean flight windows
        const seekEvents = flightTimes.filter(t => t > SEEK_THRESHOLD);

        // Backspace stats — intentionally from ALL press events, not clean.
        // These are explicit features that capture the error pattern itself.
        const totalKeys      = allPressEvents.length;
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

            // v3: seek time now from flight windows, not p2p windows
            seek_time_mean:  _mean(seekEvents),
            seek_time_count: seekEvents.length,

            shift_lag_mean:  _mean(shiftLags),
            shift_lag_std:   _std(shiftLags),
            shift_lag_count: shiftLags.length,

            extra_digraphs: Object.fromEntries(
                this.trackedDigraphs.map(dg => [dg, _mean(digraphMap[dg])])
            ),
            // 4-variant digraph timings — backend currently only consumes
            // extra_digraphs (DD).  These fields are emitted for the next
            // backend extension; safely ignored by the existing schema.
            digraph_dd_map: Object.fromEntries(
                this.trackedDigraphs.map(dg => [dg, _mean(digraphDD[dg])])
            ),
            digraph_du_map: Object.fromEntries(
                this.trackedDigraphs.map(dg => [dg, _mean(digraphDU[dg])])
            ),
            digraph_ud_map: Object.fromEntries(
                this.trackedDigraphs.map(dg => [dg, _mean(digraphUD[dg])])
            ),
            digraph_uu_map: Object.fromEntries(
                this.trackedDigraphs.map(dg => [dg, _mean(digraphUU[dg])])
            ),
            // Per-digraph flight (alias of UD) — convenient name for ML side
            flight_per_digraph: Object.fromEntries(
                this.trackedDigraphs.map(dg => [dg, _mean(digraphUD[dg])])
            ),
            // Trigraph press-to-press timings
            trigraph_map: Object.fromEntries(
                this.trackedTrigraphs.map(tg => [tg, _mean(trigraphMap[tg])])
            ),
            key_dwell_map: key_dwell_map,
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

    // ── Quality gate (TypingDNA-inspired) ────────────────────────────────────
    //
    // Returns { score: 0–1, label: string, details: {...} }
    //
    // Score is a weighted average of five sub-scores, each 0–1:
    //
    //   1. keyCount     — more keys captured = richer data (saturates at phrase length × 1.5)
    //   2. rhythm       — low coefficient of variation = consistent pace (best < 0.25)
    //   3. dwell        — mean dwell in human range (40–250ms) (penalise outliers)
    //   4. errorRate    — low backspace ratio (best = 0, worst > 0.2)
    //   5. crossAttempt — consistency vs previous attempts (only active after attempt 2)
    //
    // Weights match TypingDNA's documented quality model:
    //   keyCount 25%, rhythm 35%, dwell 20%, errorRate 10%, crossAttempt 10%
    //
    // Usage:
    //   const { score, label, details } = KeystrokeCapture.getQuality();
    //   if (score < 0.30) { /* reject */ }

    getQuality() {
        const pressEvents = this.events.filter(e => e.type === 'press');
        if (pressEvents.length < 5) {
            return { score: 0, label: 'weak', details: { reason: 'too few keystrokes' } };
        }

        const { dwellTimes, pairedPress } = this._buildDwellPairs();

        const p2pTimes = [];
        for (let i = 0; i < pressEvents.length - 1; i++) {
            const dt = pressEvents[i + 1].time - pressEvents[i].time;
            if (dt >= 0 && dt < 3000) p2pTimes.push(dt);
        }

        const phraseLen  = this.targetPhrase.replace(/\s+/g, '').length || 20;
        const totalKeys  = pressEvents.length;

        // 1. Key count score — saturates at phrase length × 1.5
        const keyCountScore = Math.min(1, totalKeys / (phraseLen * 1.5));

        // 2. Rhythm score — rhythm_cv (std / mean of p2p).  Lower = more consistent.
        //    CV < 0.25  → score 1.0
        //    CV > 0.80  → score 0.0
        //    Linear between.
        const rhythmMean = _mean(p2pTimes);
        const rhythmCv   = rhythmMean > 0 ? _std(p2pTimes) / rhythmMean : 1;
        const rhythmScore = Math.max(0, Math.min(1, (0.80 - rhythmCv) / (0.80 - 0.25)));

        // 3. Dwell score — mean dwell should be 40–250ms.
        //    Human range:  40ms (floor) → 250ms (soft ceiling).
        //    Outliers below 20ms or above 500ms penalised harshly (clamped to 0).
        const dwellMean = _mean(dwellTimes);
        let dwellScore;
        if (dwellMean < 20 || dwellMean > 500) {
            dwellScore = 0;
        } else if (dwellMean >= 40 && dwellMean <= 250) {
            dwellScore = 1;
        } else if (dwellMean < 40) {
            dwellScore = (dwellMean - 20) / 20;          // 20→40ms ramp
        } else {
            dwellScore = Math.max(0, (500 - dwellMean) / 250);  // 250→500ms ramp
        }

        // 4. Error rate score — backspace ratio 0 → 1.0, 0.2+ → 0.0
        const backRatio   = totalKeys > 0 ? this.backspaceCount / totalKeys : 0;
        const errorScore  = Math.max(0, 1 - backRatio / 0.2);

        // 5. Cross-attempt consistency (active when _prevAttempts has ≥ 1 entry)
        let crossScore = 0.75;     // neutral when no history yet
        const cv = this.getCrossAttemptCV();
        if (cv !== null) {
            // Low cv (< 0.10) = highly consistent across attempts → score 1.0
            // High cv (> 0.40) = erratic                          → score 0.0
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

    // ── Fast quality estimate (used for real-time keyup callback) ────────────
    // Skips dwell pairing (expensive) and uses only press-level metrics.

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

    // ── Cross-attempt consistency ────────────────────────────────────────────

    // Call this in enroll.js after a successful submitKeystroke(), passing the
    // features returned by extractFeatures():
    //   KeystrokeCapture.addPreviousAttempt(features);
    //
    // This lets the quality gate incorporate cross-attempt consistency from
    // attempt 2 onward — exactly what TypingDNA uses to build pattern strength.
    addPreviousAttempt(features) {
        if (!features) return;
        this._prevAttempts.push({
            typing_speed_cpm: features.typing_speed_cpm || 0,
            dwell_mean:       features.dwell_mean       || 0,
            rhythm_cv:        features.rhythm_cv        || 0,
        });
    },

    // Returns the average coefficient of variation of typing_speed_cpm and
    // dwell_mean across all previous attempts (plus the current session's
    // live metrics if available).  Returns null when < 2 data points.
    getCrossAttemptCV() {
        if (this._prevAttempts.length < 1) return null;

        const speeds = this._prevAttempts.map(a => a.typing_speed_cpm).filter(v => v > 0);
        const dwells = this._prevAttempts.map(a => a.dwell_mean).filter(v => v > 0);

        if (speeds.length < 2 && dwells.length < 2) return null;

        const cvOf = arr => {
            const m = _mean(arr);
            return m > 0 ? _std(arr) / m : 0;
        };

        const cvs = [];
        if (speeds.length >= 2) cvs.push(cvOf(speeds));
        if (dwells.length >= 2) cvs.push(cvOf(dwells));

        return cvs.length > 0 ? _mean(cvs) : null;
    },

    // ── Phrase management ────────────────────────────────────────────────────

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

        // Trigraphs: every unique [a-z]{3} window in the phrase.
        const trigSeen = new Set();
        const trigs = [];
        for (let i = 0; i < clean.length - 2; i++) {
            const tri = clean[i] + clean[i + 1] + clean[i + 2];
            if (/^[a-z]{3}$/.test(tri) && !trigSeen.has(tri)) {
                trigSeen.add(tri);
                trigs.push(tri);
            }
        }
        this.trackedTrigraphs = trigs;

        console.log(`[KeystrokeCapture] Dynamic digraphs for phrase "${phrase}":`, pairs.join(', ') || '(none)');
        console.log(`[KeystrokeCapture] Dynamic trigraphs for phrase "${phrase}":`, trigs.join(', ') || '(none)');
    },

    // ── Reset ────────────────────────────────────────────────────────────────

    // reset() clears the current attempt.  _prevAttempts is intentionally
    // NOT cleared here — it persists across attempts during enrollment.
    // Call clearHistory() explicitly if you want a full wipe (e.g. new user).
    reset() {
        this.events         = [];
        this.isCapturing    = false;
        this.startTime      = null;
        this.endTime        = null;
        this.textBuffer     = [];
        this.backspaceCount = 0;
        // Reset focus indicator if there is a currently attached input
        if (this._attachedInputId) {
            const input = document.getElementById(this._attachedInputId);
            if (input) this._updateFocusIndicator(input, false);
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