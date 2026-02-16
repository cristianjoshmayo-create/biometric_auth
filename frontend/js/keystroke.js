// frontend/js/keystroke.js
// Precision keystroke capture — core of your thesis

const KeystrokeCapture = {
    data: [],           // stores all keystroke events
    keyPressMap: {},    // tracks keydown times
    isCapturing: false,
    targetPhrase: "Biometric Voice Keystroke Authentication",

    // Attach listeners to an input element
    attach(inputElementId) {
        const input = document.getElementById(inputElementId);
        if (!input) return;

        this.data = [];
        this.keyPressMap = {};
        this.isCapturing = true;

        input.addEventListener('keydown', (e) => this.onKeyDown(e));
        input.addEventListener('keyup', (e) => this.onKeyUp(e));

        console.log("Keystroke capture attached to:", inputElementId);
    },

    onKeyDown(e) {
        if (!this.isCapturing) return;

        // performance.now() gives sub-millisecond precision
        // critical for dwell time accuracy
        this.keyPressMap[e.code] = performance.now();
    },

    onKeyUp(e) {
        if (!this.isCapturing) return;

        const keyUpTime = performance.now();
        const keyDownTime = this.keyPressMap[e.code];

        if (keyDownTime === undefined) return;

        const dwellTime = keyUpTime - keyDownTime;

        // Flight time = time between last keyUp and this keyDown
        const lastEvent = this.data[this.data.length - 1];
        const flightTime = lastEvent
            ? keyDownTime - lastEvent.keyUpTime
            : 0;

        this.data.push({
            key: e.key,
            code: e.code,
            dwellTime: parseFloat(dwellTime.toFixed(3)),
            flightTime: parseFloat(flightTime.toFixed(3)),
            keyDownTime: parseFloat(keyDownTime.toFixed(3)),
            keyUpTime: parseFloat(keyUpTime.toFixed(3))
        });

        delete this.keyPressMap[e.code];
    },

    // Extract features for the Random Forest model
    extractFeatures() {
        if (this.data.length === 0) return null;

        const dwellTimes = this.data.map(k => k.dwellTime);
        const flightTimes = this.data.map(k => k.flightTime);

        const totalTime = this.data[this.data.length - 1].keyUpTime
                        - this.data[0].keyDownTime;

        const typingSpeed = this.data.length / (totalTime / 1000); // keys per second

        return {
            dwellTimes,
            flightTimes,
            typingSpeed: parseFloat(typingSpeed.toFixed(3)),
            totalKeys: this.data.length,
            totalTime: parseFloat(totalTime.toFixed(3))
        };
    },

    // Validate the typed phrase matches target
    validatePhrase(inputValue) {
        return inputValue.trim() === this.targetPhrase;
    },

    reset() {
        this.data = [];
        this.keyPressMap = {};
        this.isCapturing = false;
    }
};