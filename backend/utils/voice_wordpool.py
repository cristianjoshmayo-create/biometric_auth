# backend/utils/voice_wordpool.py
#
# Word pool for the per-login voice challenge phrase.
#
# Why a fixed pool of common nouns/adjectives:
#   • ECAPA-TDNN is text-independent — it verifies *who* is speaking, not
#     *what* they said. So the words at login do not need to match the
#     enrollment phrase. Picking fresh words every session is safe for
#     speaker verification and turns the voice step into a strong replay
#     defense (Whisper checks word + order against the issued challenge).
#
#   • Words are common, concrete, two/three syllable, and phonetically
#     distinct so Whisper transcribes them reliably across mics and
#     accents. Homophones ("their/there", "to/two", "no/know") are excluded
#     to avoid spurious phrase-mismatch failures on legit users.
#
#   • CHALLENGE_LENGTH=4 keeps utterance length comparable to the existing
#     fixed-phrase setup. With 60 words choose 4 ordered, there are
#     ~11.6M permutations per login — large enough that an attacker with
#     prior recordings cannot enumerate or pre-record the space.
#
# Add words conservatively: every entry should transcribe cleanly through
# faster-whisper at the model size we ship. If you suspect a word is
# tripping Whisper, drop it rather than relaxing the phrase threshold.

VOICE_WORDPOOL = [
    "amber", "anchor", "autumn", "basket", "blanket",
    "branch", "breeze", "bridge", "candle", "canyon",
    "castle", "cedar", "cherry", "cloud", "copper",
    "cotton", "crystal", "daisy", "diamond", "dragon",
    "eagle", "engine", "falcon", "feather", "forest",
    "frost", "garden", "golden", "granite", "hammer",
    "harbor", "jacket", "jasmine", "jungle", "kettle",
    "lantern", "lemon", "marble", "meadow", "mirror",
    "mountain", "ocean", "orange", "pebble", "pocket",
    "purple", "quartz", "rabbit", "river", "rocket",
    "silver", "summer", "sunset", "thunder", "tunnel",
    "umbrella", "valley", "velvet", "willow", "window",
]

CHALLENGE_LENGTH = 4
